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
# using CairoMakie
# using CairoMakie: Axis
# using ColorSchemes

using CUDA
@show CUDA.has_cuda_gpu()

dev = gpu

function to_whcn(x)
	permute = x->permutedims(x,[3,2,1])
	addbatchdim = x->reshape(x, size(x)..., 1)
	x |> channelview |> permute |> addbatchdim
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
    img = rescale(img)
    img = imresize(img, 224, 224)
    arr = process(img)
    arr
end

function inspect()
    fn = sample(glob("skin-lesions/*/*/*.jpg"))
    @show fn
    img = load(fn)
    img = rescale(img)
    img = imresize(img, 224, 224)
    arr = process(img)
    @show size(arr)
    # ImageInTerminal.imshow(img)
    ImageView.imshow(img)
    arr
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
fns_train_balanced = sample(fns_train, sample_weights, length(fns_train))

# sample images for normalization
x_probe = reshape(stack(load_image.(sample(fns_train, sample_weights, 10))), 224, 224, 3, :)
std_rgb = reshape(std(x_probe, dims=(1,2,4)), 1, 1, :)
mean_rgb = reshape(mean(x_probe, dims=(1,2,4)), 1, 1, :)

# Set up model
model = ResNet(50; pretrain=true)
customres = Chain(
    backbone(model), 
    AdaptiveMeanPool((1,1)), 
    Flux.flatten, 
    Dense(2048 => 3),
    # softmax
) |> dev

# Set up optimizer and loss function
opt = Flux.setup(Adam(), customres)
loss(model, x, y) = Flux.logitcrossentropy(model(x), y)

# Use custom datastructure to load images from harddrive when
# iterated over in train loop.
struct ImageDataSource{T}
    filenames::Vector{T}
end

function nobs(ds::ImageDataSource)
    return length(ds.filenames)
end

function getobs(ds::ImageDataSource, i::Int)
    x = load_image(ds.filenames[i])[:, :, :, 1]
    x = (x .- mean_rgb)./std_rgb
    y = Float32.(get_label(ds.filenames[i]) .== labels)
    y = Flux.label_smoothing(y, 0.2f0)
    (x |> dev, y |> dev)
end
loader = DataLoader(ImageDataSource(fns_train_balanced), 8)

# Use batch for validation error
test_loader = DataLoader(ImageDataSource(fns_test), 8)
((x_test, y_test), state) = iterate(test_loader)

# training loop
for epoch in 1:50
    iter = ProgressBar(loader)
    @show Flux.logitcrossentropy(customres(x_test), y_test)
    for (x, y) in iter
        # println("Loaded a batch with $(size(x)) dims.")
        # @show y
        # yhat = customres(x) |> dev
        # @show yhat
        train!(loss, customres, [(x, y)], opt)
        yhat = customres(x)
        lv =  Flux.logitcrossentropy(yhat, y)
        set_description(iter, "Loss " * string(lv))
    end
end
@show Flux.logitcrossentropy(customres(x_test), y_test)
