using Images
using ImageInTerminal
using ImageView
using Flux
using Glob
import Base: length, getindex
using Metalhead
using StatsBase
using DataLoaders
import DataLoaders.LearnBase: getobs, nobs
using DataLoaders: DataLoader
using Random
using Flux: train!
using ProgressBars
using DataStructures
using JLD2
using Dates
using FileIO
using DataAugmentation
# using CairoMakie
# using CairoMakie: Axis
# using ColorSchemes

using CUDA
@show CUDA.has_cuda_gpu()

dev = gpu
in_width = 336
in_height = 336

function to_whcn(x)
	permute = x->permutedims(x,[3,2,1])
	addbatchdim = x->reshape(x, size(x)..., 1)
    ignore_alpha = x -> x[:, :, 1:3]
	x |> channelview |> permute |> ignore_alpha |> addbatchdim
end

collect32(x) = collect(Float32, x)

process(fn) = load(fn) |> to_whcn |> collect32
process(img::Matrix{RGB{N0f8}}) = to_whcn(img) |> collect32

function rescale(img)
    w, h = size(img)
    if w < h
        offs = rand(0:h-w)
        return img[:, 1+offs:w+offs]
    else
        offs = rand(0:w-h)
        return img[1+offs:h+offs, :]
    end
end

function load_image(fn)
    img = load(fn)
    tfm = RandomResizeCrop((in_width, in_height)) |> Maybe(FlipX()) |> Maybe(FlipY())
    img = showitems(apply(tfm, Image(img)))
    # img = rescale(img)
    # img = imresize(img, in_width, in_height)
    # arr = process(img)
    arr = img |> to_whcn |> collect32
end

function inspect()
    fn = sample(glob("skin-lesions/*/*/*.jpg"))
    @show fn
    img = load(fn)
    img = rescale(img)
    img = imresize(img, in_width, in_height)
    arr = process(img)
    @show size(arr)
    # ImageInTerminal.imshow(img)
    ImageView.imshow(img)
    arr
end

function inspect(model, fns)
    @show fn = sample(fns)
    out = fn |> load_image |> x->(x.-mean_rgb)./std_rgb |> gpu |> model |> softmax
    Dict(zip(labels, out))
end

function correct(fn)
    out = fn |> load_image |> x->(x.-mean_rgb)./std_rgb |> gpu |> model |> softmax
    labels[argmax(out)] == get_label(fn)
end


function accuracy(fns)
    mean(correct.(fns))
end


labels = ["melanoma", "nevus", "seborrheic_keratosis"]
get_label(fn) = [match.match for match in eachmatch(r"\b(melanoma|nevus|seborrheic_keratosis)\b", fn)][1]

fns_train = shuffle(glob("skin-lesions/train/*/*.jpg"))
fns_test = shuffle(glob("skin-lesions/test/*/*.jpg"))
fns_valid = shuffle(glob("skin-lesions/valid/*/*.jpg"))[1:20]

# Account for unbalanced dataset
c = counter(get_label.(fns_train))
inv_occ = Dict(l => 1 / c[l] for l in labels)
sample_weights = FrequencyWeights([inv_occ[item] for item in get_label.(fns_train)])
fns_train_balanced = sample(fns_train, sample_weights, 3*length(fns_train))

# sample images for normalization
x_probe = reshape(stack(load_image.(sample(fns_train, sample_weights, 10))), in_width, in_height, 3, :)
std_rgb = reshape(std(x_probe, dims=(1,2,4)), 1, 1, :)
mean_rgb = reshape(mean(x_probe, dims=(1,2,4)), 1, 1, :)

# Set up model
resnet = ResNet(101; pretrain=true)
customres = Chain(
    backbone(resnet), 
    AdaptiveMeanPool((1,1)), 
    Flux.flatten, 
    Dense(2048 => 3, relu),
    # Dense(2048 => 128, relu),
    # Dense(128 => 3)
    # softmax
) |> dev
# ps = Flux.params(customres[2:end])

transformer_base = ViT(:base, pretrain=false, imsize=(in_width, in_height))
transformer = Chain(
    backbone(transformer_base),
    LayerNorm(768),
    Dense(768 => 3)
) |> dev
model = customres

# Set up optimizer and loss function
opt = Flux.setup(Adam(0.001), model)
loss(model, x, y) = Flux.logitcrossentropy(model(x), y)

# Use custom datastructure to load images from harddrive when
# iterated over in train loop.
struct ImageDataSource{T}
    filenames::Vector{T}
end

function nobs(ds::ImageDataSource)
    return length(ds.filenames)
end

function getobs(ds::ImageDataSource, i::Int, gpu::Bool=true)
    x = load_image(ds.filenames[i])[:, :, :, 1]
    x = (x .- mean_rgb)./std_rgb
    y = Float32.(get_label(ds.filenames[i]) .== labels)
    y = Flux.label_smoothing(y, 0.05f0)
    if gpu
        return (x |> dev, y |> dev)
    else
        return x, y
    end
end

function get_preprocessed_data(fns)
    ds = ImageDataSource(fns)
    X = Array{Float32}(undef, in_width, in_height, 3, nobs(ds))
    Y = Array{Float32}(undef, 3, nobs(ds))
    for i in 1:nobs(ds)
        x, y = getobs(ds, i, false)
        X[:, :, :, i] = x
        Y[:, i] = y
    end
    X, Y
end

batch_size = 10
# X, Y = get_preprocessed_data(fns_train_balanced)
# FileIO.save("preprocessed.jld2", "X", X, "Y", Y)
data = FileIO.load("preprocessed.jld2")
X = data["X"]
Y = data["Y"]

# loader = DataLoader(ImageDataSource(fns_train_balanced), batch_size)
loader = DataLoader((X, Y), batch_size)

# Use batch for validation error
test_loader = DataLoader(ImageDataSource(fns_test), batch_size)
((x_test, y_test), state) = iterate(test_loader)


# training loop
function train(epochs)
    for epoch in 1:epochs
        @show epoch
        iter = ProgressBar(loader)
        @show Flux.logitcrossentropy(model(x_test), y_test)
        for (x, y) in iter
            x = x |> dev
            y = y |> dev
            # println("Loaded a batch with $(size(x)) dims.")
            # @show y
            # yhat = customres(x) |> dev
            # @show yhat
            # train!(loss, customres, [(x, y)], opt)

            grads = Flux.gradient(model) do m
                yhat = m(x)
                Flux.logitcrossentropy(yhat, y)
            end

            # Update the parameters so as to reduce the objective,
            # according the chosen optimisation rule:
            Flux.update!(opt, model, grads[1])

            yhat = model(x)
            lv =  Flux.logitcrossentropy(yhat, y)
            set_description(iter, "Loss " * string(lv))
        end
    end
    @show testloss = Flux.logitcrossentropy(model(x_test), y_test)
    jldsave("model-$(now()).jld2", model_state = Flux.state(model), loss = testloss)
end
train(20)

# using BSON
# BSON.@save "mymodel.bson" model
# BSON.@load "mymodel.bson" model
