module Cyto

using NNlib
using Zygote
using PyPlot

# generalized continuous-valued CA universe

mutable struct FloatUniverse
    grid::Array{Float64}
end


# CA universe with discrete states
mutable struct IntUniverse
    grid::Array{Int8}
end

# CA universe with binary states (e.g. Life-like CA)
mutable struct BoolUniverse
    grid::Array{Bool}
end

gaussian(u, mu, sigma) = exp.( - ((u .- mu) ./ sigma).^2 ./ 2.0 )
soft_clip(x) = 1.0 ./ (1.0 .+ exp.(-4.0 .* (x .- 0.5)))

function circular_pad(grid, padding)                                            
                                                                                
    padded = zeros(Float64, size(grid)[1]+padding*2, size(grid)[2]+padding*2)   
    padded[padding+1:end-padding, padding+1:end-padding] = grid                 
    
    padded[1:padding, padding+1:end-padding] = grid[end-padding+1:end, :]
    padded[end-padding+1:end, padding+1:end-padding] = grid[1:padding, :]
    padded[padding+1:end-padding, 1:padding] = grid[:, end-padding+1:end] 
    padded[padding+1:end-padding, end-padding+1:end] = grid[:, 1:padding]       
                                                                                                                                                                     
    return padded                                                               

end 

function get_gaussian_conv(mu::Float64=0.0, sigma::Float64=0.1, radius::Int64=5)


    rr = [sqrt( ((ii - radius - 1).^2 + (jj - radius - 1).^2)) 
            for ii = 1:radius * 2 + 1, jj=1:radius * 2 + 1 ] ./ radius

    gaussian_kernel = (rr .< radius) .* gaussian(rr, mu, sigma)

    sum_kernel = sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel ./ sum_kernel
    gaussian_kernel = reshape(gaussian_kernel, 
            (size(gaussian_kernel)[1], size(gaussian_kernel)[2], 1, 1))
  
    padding = radius 

    function gaussian_conv(grid)

        grid = circular_pad(grid[:,:,1,1], padding)
        grid = reshape(grid, (size(grid)[1], size(grid)[2], 1, 1))
        conv_grid = NNlib.conv(grid, gaussian_kernel, pad=0)

        return conv_grid
    end

    return gaussian_conv
end

function fun_neighborhood(universe::FloatUniverse, neighborhood_fun)
    
    neighborhood_grid = neighborhood_fun(universe.grid)

    return FloatUniverse(neighborhood_grid)
end


function update_fun(universe::FloatUniverse, neighborhood::FloatUniverse, params::Array{Float64})
    
    U = neighborhood.grid 
    
    temp = gaussian(U, params[2], params[3]) * 2.0 .- 1.0 

    temp = universe.grid .+ 0.1 .* temp

    temp[temp .<= 0.0] .= 0.0
    temp[temp .>= 1.0] .= 1.0

    return FloatUniverse(temp)
            
end


function fun_rules(universe::FloatUniverse, neighborhood::FloatUniverse, 
        rules_fun, params)

    new_grid = rules_fun(universe, neighborhood, params)

    return FloatUniverse(new_grid)
end

function ca_step(universe::FloatUniverse, neighborhood_fun, 
        rules_fun, params::Array{Float64}, mask_rate::Float64=0.1)

    neighborhood = fun_neighborhood(universe, neighborhood_fun)

    new_grid = rules_fun(universe, neighborhood, params).grid

    mask = rand(size(universe.grid)[1], size(universe.grid)[2], 1 , 1)

    mask = 1.0 * (mask .> mask_rate)

    #new_grid = new_grid .* (1.0 .- mask) .+ ((mask) .* universe.grid)
    universe = FloatUniverse(new_grid)

    return universe
end

end

