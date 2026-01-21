function error = compute_error(x, labels, v, w, lambda_k, beta, z_init, inputNums, outputNums)
    % 计算当前模型的误差 e_i^k(u_i)
    [y1, output] = forwardPropagation(x, v, w);
    d = zeros(size(x, 1), outputNums);
    for i = 1:length(labels)
        d(i, labels(i)+1) = 1;
    end
    deltk = output - d;
    deltj = y1 .* (1 - y1) .* (deltk * w');
    grad_v = x' * deltj;
    grad_w = y1' * deltk;
    grad_f = [grad_v(:); grad_w(:)];
    u = [v(:); w(:)];
    error = grad_f - lambda_k + beta * (u - z_init);
end

