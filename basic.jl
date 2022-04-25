# Use MCMC to do inference for basic Poisson NMF model.
# 
# Jeffrey W. Miller
# 2017-07-14 - Original version
# 2021-07-13 - Updated for compatibility with Julia 1.5.1


using Distributions
using Statistics
rmse(x,y) = sqrt.(mean((x.-y).^2, dims=1))
chisq(p,q) = sum((((p./q) .- 1).^2).*q, dims=1)

# -------------------- Main functions --------------------


# Generate simulated data
function generate_basic_data(I,J,K,normalized)
    a = 1
    alpha = 0.5
    #b = rand(Gamma(2,1/2),K)
    b = ones(K)/1 # ones(K)/4
    t = rand(Gamma(a,1),J,K) ./ b'
    r = rand(Gamma(alpha,1),I,K) ./ b'
    if normalized
        t = t.*sum(r,dims=1)
        r = r./sum(r,dims=1)
    end
    l = r*t'
    x = Int[rand(Poisson(l[i,j])) for i=1:I, j=1:J]
    return x,r,t,b
end

# Generate simulated data
function generate_basic_data_alt(I,J,K,normalized)
    a = 1
    alpha = 0.5
    if normalized
        #mu = (25*I/K)*ones(K)
        mu = rand(Gamma(50,1/50),K) * 400
        t = rand(Gamma(a,1/a),J,K) .* mu'
        r = rand(Gamma(alpha,1),I,K)
        r = r./sum(r,dims=1)
    else
        #mu = sqrt(25/K)*ones(K)
        mu = rand(Gamma(50,1/50),K) * 4
        t = rand(Gamma(a,1/a),J,K) .* mu'
        r = rand(Gamma(alpha,1/alpha),I,K) .* mu'
    end
    l = r*t'
    x = Int[rand(Poisson(l[i,j])) for i=1:I, j=1:J]
    return x,r,t
end

# MCMC sampler for basic Poisson NMF model
function basic_sampler(x,K,a0,b0; n_samples=1000, normalized=true, a=1, alpha=0.5, r=Float64[], zeta=1.0)
    I,J = size(x)

    # Initialize
    mu = rand(InverseGamma(a0,b0),K)
    t = rand(Gamma(a,1/a),J,K) .* mu'
    y = zeros(Float64,I,J,K)
    if isempty(r)
        r = rand(Gamma(alpha,1/alpha),I,K) .* mu'
        infer_r = true
    else
        K = size(r,2)
        infer_r = false
    end

    if normalized; r = r./sum(r,dims=1); end

    # Record-keeping
    rr = zeros(Float64,n_samples,I,K)
    tr = zeros(Float64,n_samples,J,K)
    mr = zeros(Float64,n_samples,K)

    # Run
    for iter = 1:n_samples
        # Update y
        for i = 1:I, j = 1:J
            q = r[i,:].*t[j,:]
            y[i,j,:] = rand(Multinomial(x[i,j],q/sum(q)))
        end

        # Update t
        for j = 1:J, k = 1:K
            A = a + zeta*sum(y[:,j,k])
            B = a/mu[k] + zeta*sum(r[:,k])
            t[j,k] = rand(Gamma(A,1/B))
        end
        
        # Update r
        if infer_r
            if normalized
                for k = 1:K
                    A = alpha .+ zeta*vec(sum(y[:,:,k],dims=2))
                    r[:,k] = rand(Dirichlet(A))
                end
            else
                for i = 1:I, k = 1:K
                    A = alpha + zeta*sum(y[i,:,k])
                    B = alpha/mu[k] + zeta*sum(t[:,k])
                    r[i,k] = rand(Gamma(A,1/B))
                end
            end
        end

        # Update hyperparameters
        if normalized
            A = a0 + J*a
            B = b0 .+ a*vec(sum(t,dims=1))
            mu = rand(InverseGamma(A,1),K) .* B
        else
            A = a0 + I*alpha + J*a
            B = b0 .+ alpha*vec(sum(r,dims=1)) + a*vec(sum(t,dims=1))
            mu = rand(InverseGamma(A,1),K) .* B
        end

        if (iter%10)==0
            println(iter,": mu=",round.(mu,digits=1))
        end

        # Record
        rr[iter,:,:] = r
        tr[iter,:,:] = t
        mr[iter,:] = mu
    end
    return rr,tr,mr
end

# ARD-NMF using a modified Tan-Fevotte algorithm (https://arxiv.org/pdf/1111.6085.pdf)
function ARD_NMF(V,k,a; maxiter=1000, convtol=1e-8, verbose=false, w1unif=false, wprob=false, 
                        W=rand(size(V,1),k), H=rand(k,size(V,2)), value=Float64[])
    n,m = size(V)
    # W = rand(n,k)
    # H = rand(k,m)
    if w1unif; W[:,1] = 1/n; end
    if wprob; W = W./sum(W,dims=1); end
    c = n+m+a+1
    b = sqrt((a-1)*(a-2)*mean(V)/k)
    phi = 1
    if isempty(value)
        lambda = rand(InverseGamma(a,b),k)
    else
        lambda = value.*(b/c) .+ b/c
    end
    lambda_prev = copy(lambda)
    for iter = 1:maxiter
        H = H.*((W'*(V./(W*H .+ eps())))./(W'*ones(n,m) .+ phi./(lambda .+ eps()) .+ eps()))
        W = W.*(((V./(W*H .+ eps()))*H')./(ones(n,m)*H' .+ phi./(lambda' .+ eps()) .+ eps()))
        if w1unif; W[:,1] .= 1/n; end
        if wprob; W = W./(sum(W,dims=1).+eps()); end
        lambda = (vec(sum(W,dims=1)) + vec(sum(H,dims=2)) .+ b)./c
        if maximum(abs.((lambda - lambda_prev)./lambda_prev)) < convtol; break; end
        lambda_prev = copy(lambda)
        if verbose; println(lambda); end
        if iter==maxiter; @warn("Maximum number of iterations reached. Possible nonconvergence."); end
    end
    value = (lambda .- b/c)./(b/c)
    if any(isnan.(W)) || any(isnan.(H))
        @warn("Got NaNs in ARD_NMF. Trying again.")
        return ARD_NMF(V,k,a; maxiter=maxiter, convtol=convtol, verbose=verbose, w1unif=w1unif, wprob=wprob)
    else
        return W,H,value
    end
end

# Bootstrap ARD-NMF
function bootstrap_ARD_NMF(x,K,a,n_samples; kwargs...)
    # Initialize
    I,J = size(x)
    r = rand(I,K)
    t = rand(J,K)
    m = Float64[]
    tt = t'
    
    # Record-keeping
    rr = zeros(Float64,n_samples,I,K)
    tr = zeros(Float64,n_samples,J,K)
    mr = zeros(Float64,n_samples,K)

    for iter=1:n_samples
        r,tt,m = ARD_NMF(x,K,a; W=r, H=tt, value=m, kwargs...)
        rr[iter,:,:] = r
        tr[iter,:,:] = tt'
        mr[iter,:] = m
    end
    return rr,tr,mr
end

# Try (greedily) to find the correct permutation of the columns to align A with B
function permute_to_match(A,B,dist)
    @assert(size(A)==size(B))
    K = size(A,2)
    remaining = collect(1:K)
    order = Int[]
    for kb = 1:K
        #dists = sum((B[:,kb] .- A[:,remaining]).^2, dims=1)
        dists = vec(dist(B[:,kb],A[:,remaining]))
        k = remaining[argmin(dists)]
        push!(order,k)
        remaining = setdiff(remaining,k)
    end
    return order
end


function estimate(rs,ts,ms,tol)
    n = size(rs,1)

    # handle possible label switching
    #rn = rs[n,:,:]
    #for i = 1:n-1
        #order = permute_to_match(rs[i,:,:],rn,chisq)
        #rs[i,:,:] = rs[i,:,order]
        #ts[i,:,:] = ts[i,:,order]
        #ms[i,:] = ms[i,order]
    #end

    # normalize
    for i = 1:n
        ts[i,:,:] = ts[i,:,:] .* sum(rs[i,:,:],dims=1)
        rs[i,:,:] = rs[i,:,:] ./ sum(rs[i,:,:],dims=1)
    end

    # average
    r = dropdims(mean(rs,dims=1),dims=1)
    t = dropdims(mean(ts,dims=1),dims=1)
    mu = vec(mean(ms,dims=1))

    # subset
    subset = findall(mu.>tol)
    return r[:,subset],t[:,subset]
end

function compare(A,B,dist,padding)
    I,KA = size(A)
    IB,KB = size(B)
    # pad to match number of columns
    @assert(I==IB)
    if KB > KA # pad A if too few vectors
        @warn("Padding A with $(KB-KA) vectors to facilitate comparison.")
        A = [A padding*ones(I,KB-KA)]
    elseif KB < KA # pad B if too many vectors
        @warn("Padding B with $(KA-KB) vectors to facilitate comparison.")
        B = [B padding*ones(I,KA-KB)]
    end
    # permute columns to align
    order = permute_to_match(A,B,dist)
    # compute mean distance
    return mean(dist(A[:,order],B))
end


# -------------------- Test on data from the assumed model --------------------
function run_example()
    # settings
    I,J,K0,K = 96,100,5,10
    #I,J,K0,K = 96,200,10,20
    # I,J,K0,K = 96,400,20,30
    nreps = 10 # 10
    normalized = true
    n_total = 2000 # 2000
    n_burn = 1000 # 1000
    use_true_r = true

    # model parameters
    a = 1
    alpha = 0.5
    epsilon = 0.01 # 1e-4  # ARD tolerance param
    tol = 0.04

    # record-keeping
    K_s = zeros(Int,nreps,2)
    chisq_r_s = zeros(nreps,2)
    chisq_r1_s = zeros(nreps,2)
    rmse_r_s = zeros(nreps,2)
    rmse_t_s = zeros(nreps,2)
    times = zeros(nreps,2)

    for rep = 1:nreps
        # generate data
        x,r0,t0 = generate_basic_data(I,J,K0,normalized)

        # run sampler
        a0 = J*a+1
        b0 = epsilon*(a0-1)
        start_time = time()
        r = (use_true_r ? r0 : Float64[])
        rr,tr,mr = basic_sampler(x,K,a0,b0; n_samples=n_total, normalized=normalized, a=a, alpha=alpha, r=r)
        times[rep,1] = time() - start_time

        # estimate normalized r and t
        use = n_burn+1:n_total
        r_hat,t_hat = estimate(rr[use,:,:],tr[use,:,:],mr[use,:],tol)
        K_s[rep,1] = K_hat = size(r_hat,2)
        t0n = t0.*sum(r0,dims=1)
        r0n = r0./sum(r0,dims=1)

        # compare with true r and t
        chisq_r_s[rep,1] = chisq_r = compare(r_hat,r0n,chisq,1/I)
        chisq_r1_s[rep,1] = chisq_r1 = compare(r0n,r_hat,chisq,1/I)
        rmse_r_s[rep,1] = rmse_r = compare(r_hat,r0n,rmse,0)
        rmse_t_s[rep,1] = rmse_t = compare(t_hat,t0n,rmse,0)
        println("K_hat = ",K_hat)
        println("K0 = ",K0)
        println("ChiSq(r,r0) = ",chisq_r)
        println("ChiSq(r0,r) = ",chisq_r1)
        println("RMSE(r,r0) = ",rmse_r)
        println("RMSE(t,t0) = ",rmse_t)

        # compare ARD-NMF with true r and t
        start_time = time()
        W,H,value = ARD_NMF(x,K,5; maxiter=10000, convtol=1e-6, verbose=false, w1unif=false, wprob=normalized)
        times[rep,2] = time() - start_time
        subset = findall(value.>1)
        r_TF,t_TF = W[:,subset],H[subset,:]'
        t_TF = t_TF.*sum(r_TF,dims=1)
        r_TF = r_TF./sum(r_TF,dims=1)
        K_s[rep,2] = K_hat_TF = length(subset)
        chisq_r_s[rep,2] = chisq_r_TF = compare(r_TF,r0n,chisq,1/I)
        chisq_r1_s[rep,2] = chisq_r1_TF = compare(r0n,r_TF,chisq,1/I)
        rmse_r_s[rep,2] = rmse_r_TF = compare(r_TF,r0n,rmse,0)
        rmse_t_s[rep,2] = rmse_t_TF = compare(t_TF,t0n,rmse,0)
        println("K_hat_TF = ",K_hat_TF)
        println("ChiSq(r_TF,r0) = ",chisq_r_TF)
        println("ChiSq(r0,r_TF) = ",chisq_r1_TF)
        println("RMSE(r_TF,r0) = ",rmse_r_TF)
        println("RMSE(t_TF,t0) = ",rmse_t_TF)
    end

    println("K_s = ")
    display(K_s')
    println("chisq_r_s = ")
    display(chisq_r_s')
    println("means = ",mean(chisq_r_s,dims=1))
    println("chisq_r1_s = ")
    display(chisq_r1_s')
    println("means = ",mean(chisq_r1_s,dims=1))
    println("rmse_r_s = ")
    display(rmse_r_s')
    println("means = ",mean(rmse_r_s,dims=1))
    println("rmse_t_s = ")
    display(rmse_t_s')
    println("means = ",mean(rmse_t_s,dims=1))
    println("times = ")
    display(times')
    println("means = ",mean(times,dims=1))
end









