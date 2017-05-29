function [F_LM, cost_lst] = levenbergEst(F_init, inlier1, inlier2, pnt3D)
    F = F_init;
    cost_lst = [];
    inlier1 = inlier1(:, 1 : 2);
    inlier2 = inlier2(:, 1 : 2);
    inlier1 = inlier1';
    inlier2 = inlier2';
    pnt3D = pnt3D';
    num_pnt = size(inlier1, 2);
    Z = [0, -1, 0;
         1, 0, 0;
         0, 0, 1];
    
    % normarlize 3D points
    for i = 1 : num_pnt
        pnt3D(:, i) = pnt3D(:, i) / norm(pnt3D(:, i));
    end
    
    % compute initial P and P_prime
    P = [eye(3), zeros(3, 1)];
    [w_u, w_v, sigma, s] = parameterize_F(F);
    [U, D, V] = svd(F);
    D_prime = D;
    D_prime(3, 3) = (D(1, 1) + D(2, 2)) / 2.0;
    m = U * Z * D_prime * V';
    e_prime = -U(:, 3);
    P_prime = [m, e_prime];
    
    % compute initial cost
    pnt_pred1 = P * pnt3D;
    pnt_pred1 = pnt_pred1 ./ pnt_pred1(3, :);
    pnt_pred1_inhomo = pnt_pred1(1 : 2, :);
    pnt_pred2 = P_prime * pnt3D;
    pnt_pred2 = pnt_pred2 ./ pnt_pred2(3, :);
    pnt_pred2_inhomo = pnt_pred2(1 : 2, :);
    error1 = pnt_pred1_inhomo - inlier1;
    error2 = pnt_pred2_inhomo - inlier2;
    cost = norm(error1)^2 + norm(error2)^2;
    cost_lst = [cost_lst, cost];
    
    % parameterization of 3D point
    pnt_scene_param = zeros(3, num_pnt);
    for i = 1 : num_pnt
        pnt = pnt3D(:, i);
        pnt_param = parameterize(pnt);
        pnt_scene_param(:, i) = pnt_param;
    end
    
    param_F = [w_u', w_v', s]';
    lam = 0.001;
    n = 0;
    % begin iteration
    for k = 1 : 40
        % compute Jacobian matrix
        [A, B1, B2] = calJacobian(num_pnt, pnt_pred1, pnt_pred2, pnt_scene_param, P_prime, param_F);
        
        % compute normal equations matrix
        U = zeros(7, 7);
        V = zeros(3, 3, num_pnt);
        W = zeros(7, 3, num_pnt);
        error_part1 = zeros(7, 1);
        error_part2 = zeros(3, 1, num_pnt);
        for i = 1 : num_pnt
            U = U + A(:, :, i)' * A(:, :, i);
            V(:, :, i) = B1(:, :, i)' * B1(:, :, i) + ...
                         B2(:, :, i)' * B2(:, :, i);
            W(:, :, i) = A(:, :, i)' * B2(:, :, i);
            error_part1 = error_part1 + A(:, :, i)' * error2(:, i);
            error_part2(:, :, i) = B1(:, :, i)' * error1(:, i) + ...
                                   B2(:, :, i)' * error2(:, i);
        end

        % compute augmented normal euqations
        S = U + lam * eye(7);
        epsilon = error_part1;
        for i = 1 : num_pnt
            S = S - W(:, :, i) * inv(V(:, :, i) + lam * eye(3)) * W(:, :, i)';
            epsilon = epsilon - W(:, :, i) * inv(V(:, :, i) + lam * eye(3)) * error_part2(:, :, i);
        end
        delata_part1 = linsolve(S, epsilon);
        delata_part2 = zeros(3, 1, num_pnt);
        for i = 1 : num_pnt
            delata_part2(:, :, i) = inv(V(:, :, i) + lam * eye(3)) * ...
                                    (error_part2(:, :, i) - W(:, :, i)' * delata_part1);
        end

        % update
        param_F_update = param_F + delata_part1;
        %{
        w_u_update = param_F_update(1 : 3);
        w_v_update = param_F_update(4 : 6);
        s_update = param_F_update(7);
        sigma_update = deparameterize(s_update);
        F_update = expm(formMat(w_u_update)) * diag([sigma_update', 0]) * expm(formMat(w_v_update))';
        %}
        F_update = deparameterize_F(param_F_update(1 : 3), param_F_update(4 : 6), param_F_update(7));
        pnt_scene_param_update = zeros(3, num_pnt);
        pnt3D_update = zeros(4, num_pnt);
        for i = 1 : num_pnt
            pnt_scene_param_update(:, i) = pnt_scene_param(:, i) + delata_part2(:, :, i);
            pnt3D_update(:, i) = deparameterize(pnt_scene_param_update(:, i));
            pnt3D_update(:, i) = pnt3D_update(:, i) / pnt3D_update(4, i);
            pnt3D_update(:, i) = pnt3D_update(:, i) / norm(pnt3D_update(:, i));
        end
        
        % compute P, P_prime and cost
        P_update = [eye(3), zeros(3, 1)];
        P_prime_update = cal_P_prime(F_update);
        [cost_update, error1_update, error2_update, pnt_pred1_update, pnt_pred2_update] = ...
            cal_cost(P_update, P_prime_update, pnt3D_update, inlier1, inlier2);

        % jump the loop
        if(cost_update > cost)
            lam = 10 * lam;
        else
            n = n + 1;
            lam = lam / 10;
            param_F = param_F_update;
            F = F_update;
            P_prime = P_prime_update;
            error1 = error1_update;
            error2 = error2_update;
            cost = cost_update;
            cost_lst = [cost_lst, cost];
            pnt_scene_param = pnt_scene_param_update; 
            pnt3D = pnt3D_update;
            pnt_pred1 = pnt_pred1_update;
            pnt_pred2 = pnt_pred2_update;
        end
    end
    F_LM = F;
end

% parameteriztion of matrix F
function [w_u, w_v, sigma, s] = parameterize_F(F)
    [U, D, V] = svd(F);
    if det(U) < 0
        U = -U;
    end
    if det(V) < 0
        V = -V;
    end
    D = [D(1, 1), D(2, 2)]';
    D = D / norm(D);
    tmp_u = logm(U);
    w_u = [tmp_u(3, 2), tmp_u(1, 3), tmp_u(2, 1)]';
    tmp_v = logm(V);
    w_v = [tmp_v(3, 2), tmp_v(1, 3), tmp_v(2, 1)]';
    sigma = D;
    s = parameterize(sigma);
end

% deparameterization of matrix F
function F = deparameterize_F(w_u, w_v, s)
    U = expm(formMat(w_u));
    V = expm(formMat(w_v));
    sigma = deparameterize(s);
    F = sigma(1) * U(:, 1) * V(:, 1)' + sigma(2) * U(:, 2) * V(:, 2)';
end

% parameterize
function paramVector = parameterize(P)
    a = P(1);
    b = P(2 : length(P));
    paramVector = (2.0 / (sinc(acos(a)))) * b;
    normP = norm(paramVector);
    if (normP > pi)
        paramVector = (1.0 - 2 * pi / normP * ceil((normP - pi) / 2 * pi)) * paramVector;
    end
end

% deparameterize
function deparamVector = deparameterize(P)
    normP = norm(P);
    deparamVector = [cos(normP / 2.0), ((sinc(normP / 2.0)) / 2.0) * P']';
end

% sinc(x)
function res = sinc(x)
    if x == 0
        res = 1.0;
    else
        res = (sin(x)) / x;
    end
end

% derivative of sinc(x)
function res = derivSinc(x)
    if x == 0
        res = 0.0;
    else
        res = cos(x) / x - sin(x) / (x * x);
    end
end

% form the matrix
function mat = formMat(X)
    mat = [0, -X(3), X(2);...
           X(3), 0, -X(1);...
           -X(2), X(1), 0];
end

% compute Jacobian matrix
function [A, B1, B2] = calJacobian(num_pnt, pnt_pred1, pnt_pred2, pnt_scene_param, P_prime, param_F)
    w_u = param_F(1 : 3);
    w_v = param_F(4 : 6);
    s = param_F(7);
    sigma = deparameterize(s);
    A = zeros(2, 7, num_pnt);
    B1 = zeros(2, 3, num_pnt);
    B2 = zeros(2, 3, num_pnt);
    for i = 1:num_pnt
        pnt1 = pnt_pred1(:, i);
        pnt2 = pnt_pred2(:, i);
        pnt_scene = deparameterize(pnt_scene_param(:, i));

        % compute A
        A1_i = [1 / pnt2(3), 0, -pnt2(1) / pnt2(3)^2;
                0, 1 / pnt2(3), -pnt2(2) / pnt2(3)^2];
        A2_i = zeros(3, 12);
        A2_i(1, 1 : 4) = pnt_scene';
        A2_i(2, 5 : 8) = pnt_scene';
        A2_i(3, 9 : 12) = pnt_scene';
        
        U = expm(formMat(w_u));
        V = expm(formMat(w_v));
        tmp_mat = zeros(9, 3);
        tmp_mat(2, 3) = -1;
        tmp_mat(3, 2) = 1;
        tmp_mat(4, 3) = 1;
        tmp_mat(6, 1) = -1;
        tmp_mat(7, 2) = -1;
        tmp_mat(8, 1) = 1;
        
        % compute dP' / dw_u
        tmp = [-sigma(2) * V(:, 2), sigma(1) * V(:, 1), (sigma(1) + sigma(2)) / 2.0 * V(:, 3);
               0, 0, -1];
        dp_du = kron(eye(3), tmp);        
        theta = norm(w_u);
        dtheta_dw = (1.0 / theta) * w_u';
        s = (1 - cos(theta) / theta^2);
        tmp_m = vec(w_u * w_u');
        dm_dw = kron(w_u, eye(3)) + kron(eye(3), w_u);
        ds_dw = dtheta_dw * ((theta * sin(theta) - 2 * (1 - cos(theta))) / theta^3);
        du_dw_u = -vec(eye(3)) * sin(theta) * dtheta_dw + sinc(theta) * tmp_mat + ...
                  vec(formMat(w_u)) * derivSinc(theta) * dtheta_dw + s * dm_dw + tmp_m * ds_dw;
        dp_dw_u = dp_du * du_dw_u;
        
        % compute dP' / dw_v
        dp_dv = [kron(eye(3), [sigma(1) * U(1, 2), -sigma(2) * U(1, 1), (sigma(1) + sigma(2)) / 2.0 * U(1, 3)]);
                 zeros(1, 9);
                 kron(eye(3), [sigma(1) * U(2, 2), -sigma(2) * U(2, 1), (sigma(1) + sigma(2)) / 2.0 * U(2, 3)]);
                 zeros(1, 9);
                 kron(eye(3), [sigma(1) * U(3, 2), -sigma(2) * U(3, 1), (sigma(1) + sigma(2)) / 2.0 * U(3, 3)]);
                 zeros(1, 9)];    
        theta = norm(w_v);
        dtheta_dw = (1.0 / theta) * w_v';
        s = (1 - cos(theta) / theta^2);
        tmp_m = vec(w_v * w_v');
        dm_dw = kron(w_v, eye(3)) + kron(eye(3), w_v);
        ds_dw = dtheta_dw * ((theta * sin(theta) - 2 * (1 - cos(theta))) / theta^3);
        dv_dw_v = -vec(eye(3)) * sin(theta) * dtheta_dw + sinc(theta) * tmp_mat + ...
                  vec(formMat(w_v)) * derivSinc(theta) * dtheta_dw + s * dm_dw + tmp_m * ds_dw;
        dp_dw_v = dp_dv * dv_dw_v;
        
        % compute dP' / ds
        dp_dsigma = [U(1, 2) * V(:, 1) + 0.5 * U(1, 3) * V(:, 3), 0.5 * U(1, 3) * V(:, 3) - U(1, 1) * V(:, 2);
                     0, 0;
                     U(2, 2) * V(:, 1) + 0.5 * U(2, 3) * V(:, 3), 0.5 * U(2, 3) * V(:, 3) - U(2, 1) * V(:, 2);
                     0, 0;
                     U(3, 2) * V(:, 1) + 0.5 * U(3, 3) * V(:, 3), 0.5 * U(3, 3) * V(:, 3) - U(3, 1) * V(:, 2);
                     0, 0];
        dsigma_ds = zeros(2, 1);
        dsigma_ds(1) = -0.5 * sigma(2);
        if norm(s) == 0
            dsigma_ds(2) = 0.5;
        else
            dsigma_ds(2) = 0.5 * sinc(0.5 * norm(s)) + 0.25 * norm(s) * derivSinc(0.5 * norm(s)) * s * s;
        end
        dp_ds = dp_dsigma * dsigma_ds;
        
        A3_i = [dp_dw_u, dp_dw_v, dp_ds];
        A_i = A1_i * A2_i * A3_i;
        A(:, :, i) = A_i;

        % compute B1
        B1_1_i = [1 / pnt1(3), 0, -pnt1(1) / pnt1(3)^2;
                  0, 1 / pnt1(3), -pnt1(2) / pnt1(3)^2];
        B1_2_i = [eye(3), zeros(3, 1)];
        B1_3_i = zeros(4, 3);
        B1_3_i(1, :) = -0.25 * (sinc(norm(pnt_scene_param(:, i)) / 2)) * ...
                       pnt_scene_param(:, i)';
        if norm(pnt_scene_param(:, i)) == 0
            B1_3_i(2 : 4, :) = 0.5 * eye(3);
        else
            B1_3_i(2 : 4, :) = sinc(norm(pnt_scene_param(:, i)) / 2) * 0.5 * ...
                               eye(3) + (1/(4 * norm(pnt_scene_param(:, i)))) * ...
                               (derivSinc(norm(pnt_scene_param(:, i)) / 2)) * ...
                               pnt_scene_param(:, i) * pnt_scene_param(:, i)';
        end
        B1_i = B1_1_i * B1_2_i * B1_3_i;
        B1(:, :, i) = B1_i;

        % compute B2
        B2_1_i = [1 / pnt2(3), 0, -pnt2(1) / pnt2(3)^2;
                  0, 1 / pnt2(3), -pnt2(2) / pnt2(3)^2];

        B2_2_i = P_prime;

        B2_3_i = zeros(4, 3);
        B2_3_i(1, :) = -0.25 * (sinc(norm(pnt_scene_param(:, i)) / 2)) * ...
                      pnt_scene_param(:, i)';
        if norm(pnt_scene_param(:, i)) == 0
            B2_3_i(2 : 4, :) = 0.5 * eye(3);
        else
            B2_3_i(2 : 4, :) = sinc(norm(pnt_scene_param(:, i)) / 2) * 0.5 * ...
                               eye(3) + (1/(4 * norm(pnt_scene_param(:, i)))) * ...
                               (derivSinc(norm(pnt_scene_param(:, i)) / 2)) * ...
                               pnt_scene_param(:, i) * pnt_scene_param(:, i)';
        end
        B2_i = B2_1_i * B2_2_i * B2_3_i;
        B2(:, :, i) = B2_i;
    end
end

% vectorize
function res = vec(a)
    tmp = a';
    res = tmp(:);
end

% compute P_prime
function P_prime = cal_P_prime(F)
    Z = [0, -1, 0;
         1, 0, 0;
         0, 0, 1];
    [U, D, V] = svd(F);
    D_prime = D;
    D_prime(3, 3) = (D(1, 1) + D(2, 2)) / 2.0;
    m = U * Z * D_prime * V';
    e_prime = -U(:, 3);
    P_prime = [m, e_prime];
end

% compute cost
function [cost, error1, error2, pnt_pred1, pnt_pred2] = cal_cost(P, P_prime, pnt3D, inlier1, inlier2)
    pnt_pred1 = P * pnt3D;
    pnt_pred1 = pnt_pred1 ./ pnt_pred1(3, :);
    pnt_pred1_inhomo = pnt_pred1(1 : 2, :);
    pnt_pred2 = P_prime * pnt3D;
    pnt_pred2 = pnt_pred2 ./ pnt_pred2(3, :);
    pnt_pred2_inhomo = pnt_pred2(1 : 2, :);
    error1 = pnt_pred1_inhomo - inlier1;
    error2 = pnt_pred2_inhomo - inlier2;
    cost = norm(error1)^2 + norm(error2)^2;
end