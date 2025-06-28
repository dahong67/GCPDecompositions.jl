## Script to create the logo file `docs/src/assets/logo.svg`

# Activate and instantiate the `docs` environment
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Load Luxor
using Luxor

# Setup path
ASSET_DIR = joinpath(@__DIR__, "src", "assets")
mkpath(ASSET_DIR)

# Setup drawing
Drawing(500, 500, joinpath(ASSET_DIR, "logo.svg"))
setline(4)

## Draw three-way outer product

# Draw mode-1 factor
origin(Point(25, 25 + 40 * 4 + 10))
for i in 0:6
    poly(rect(Point(0, 40 * i), 40, 40; vertices=true); close=true)
    setcolor(Luxor.julia_red)
    fillpreserve()
    sethue("black")
    strokepath()
end

# Draw mode-2 factor
origin(Point(25 + 40 + 10, 25 + 40 * 4 + 10))
for i in 0:9
    poly(rect(Point(40 * i, 0), 40, 40; vertices=true); close=true)
    setcolor(Luxor.julia_purple)
    fillpreserve()
    sethue("black")
    strokepath()
end

# Draw mode-3 factor
origin(Point(25, 25 + 40 * 4))
transform([1 0 0.4 -0.8 0 0])
for i in 0:4
    poly(rect(Point(0, 40 * i), 40, 40; vertices=true); close=true)
    setcolor(Luxor.julia_green)
    fillpreserve()
    sethue("black")
    strokepath()
end

## Draw loss functions
using Distributions
origin(Point(25 + 40 + 10 + 100, 175))
Luxor.scale(30, -150)
setline(8)
for (dist, color) in [
    Gamma() => Luxor.julia_green,
    Chisq(3) => Luxor.julia_blue,
    Rayleigh() => Luxor.julia_red,
]
    sethue(color)
    poly([Point(x, pdf(dist,x)) for x in 0:0.1:10], :stroke)
end
sethue("black")
line(Point(0, 0), Point(10, 0); action = :stroke)

# ## Draw heterogeneous matrix
# using StableRNGs
# rng = StableRNG(1)
# origin(Point(25 + 10 + 20 * 4, 25 + 10 + 20 * 12))
# setline(4)
# for i in 0:17, j in 0:9
#     poly(rect(Point(20 * i, 20 * j), 20, 20; vertices=true); close=true)
#     setcolor(rand(rng, [Luxor.julia_red, Luxor.julia_green, Luxor.julia_blue, Luxor.julia_purple]))
#     fillpreserve()
#     sethue("black")
#     strokepath()
# end

## Draw factor viz
using StableRNGs
rng = StableRNG(4)
for i in 0:2, j in 0:1
    origin(Point(
        25 + 40 + 10 + 20 + 125 * i,
        25 + 260 + 40 + 30 + 100 * j,
    ))
    Luxor.scale(115, -90)
    setline(4)

    # Axes background
    setline(4)
    poly(rect(Point(0, 0), 1, 1; vertices=true); close=true)
    sethue("white")
    fillpreserve()

    # Factors
    if i == 0    # mode 1
        sethue(Luxor.julia_purple)
        setline(4)
        poly([Point(x, 1/2+sin(x*(j+1)*2*pi)/3) for x in 0:0.01:1], :stroke)
    elseif i == 1    # mode 2
        setline(0.5)
        for x in 0:1/8:7/8
            poly(rect(Point(x, 0), 1/8, rand(rng); vertices=true); close=true)
            sethue(Luxor.julia_green)
            fillpreserve()
            sethue("white")
            strokepath()
        end
    elseif i == 2    # mode 3
        for x in 0.1:0.1:0.9
            y = 0.8 * rand(rng) + 0.1
            sethue(y > 0.5 ? Luxor.julia_red : Luxor.julia_blue)
            circle(Point(x, y), 0.05; action = :fill)
        end
    end

    # Axes border
    setline(4)
    poly(rect(Point(0, 0), 1, 1; vertices=true); close=true)
    sethue("black")
    strokepath()
end

# Finish drawing
finish()
preview()