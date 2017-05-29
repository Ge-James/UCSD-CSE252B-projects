function H_LM = levenbergEst(H_init, inlier1, inlier2)
    inlier1 = inlier1';
    inlier2 = inlier2';
    num_point = size(inlier1, 2);
    pntHomo1 = [inlier1; ones(1, num_point)];
    pntHomo2 = [inlier2; ones(1, num_point)];
    H = H_init;
    
    % compute scene point
    pnt_scene_deparam = zeros(2, num_point);
    pnt_scene_param = zeros(2, num_point);
    for i = 1 : num_point
       pnt = sampsonCorrection(H, pntHomo1(:, i)', pntHomo2(:, i)');
       pnt = pnt';
       pnt_scene_deparam(:, i) = pnt;
       pnt = [pnt; 1] / norm(pnt);
       pnt_scene_param(:, i) = parameterize(pnt);
    end

    % compute initial cost
    error1 = inlier1 - pnt_scene_deparam;
    proj2 = zeros(3, num_point);
    proj2_homo = zeros(3, num_point);
    for i = 1:num_point
       proj2(:, i) = H * deparameterize(pnt_scene_param(:, i));
       proj2_homo(:, i) = proj2(:, i)/proj2(3, i); 
    end
    proj2_inhomo = proj2_homo(1:2, :);
    error2 = inlier2 - proj2_inhomo;
    cost = norm(error1(:))^2 + norm(error2(:))^2;
    disp('cost for each iteration:');
    disp(cost);

    h = H';
    h = h(:);
    paramH = parameterize(h);
    lam = 0.001;
    n = 1;

    while 1
        % compute Jacobian matrix
        [A, B1, B2] = calJacobian(num_point, proj2, pnt_scene_param, H, paramH);

        % compute normal equations matrix
        U = zeros(8, 8);
        V = zeros(2, 2, num_point);
        W = zeros(8, 2, num_point);
        error_part1 = zeros(8, 1);
        error_part2 = zeros(2, 1, num_point);
        for i = 1:num_point
            U = U + A(:, :, i)' * A(:, :, i);
            V(:, :, i) = B1(:, :, i)' * B1(:, :, i) + ...
                         B2(:, :, i)' * B2(:, :, i);
            W(:, :, i) = A(:, :, i)' * B2(:, :, i);
            error_part1 = error_part1 + A(:, :, i)' * ...
                          error2(:, i);
            error_part2(:, :, i) = B1(:, :, i)' * error1(:, i) + ...
                                   B2(:, :, i)' * error2(:, i);
        end

        % compute augmented normal euqations
        S = U + lam * eye(8);
        epsilon = error_part1;
        for i = 1 : num_point
            S = S - W(:, :, i) * inv(V(:, :, i) + lam * eye(2)) * ...
                W(:, :, i)';
            epsilon = epsilon - W(:, :, i) * inv(V(:, :, i) + lam * eye(2)) * ...
                      error_part2(:, :, i);
        end
        delata_part1 = linsolve(S, epsilon);
        delata_part2 = zeros(2, 1, num_point);
        for i = 1 : num_point
            delata_part2(:, :, i) = inv(V(:, :, i) + lam * eye(2)) * ...
                                    (error_part2(:, :, i) - W(:, :, i)' * ...
                                    delata_part1);
        end

        % update
        paramH_update = paramH + delata_part1;
        pnt_scene_param_update = zeros(2, num_point);
        for i = 1 : num_point
            pnt_scene_param_update(:, i) = pnt_scene_param(:, i) + ...
                                           delata_part2(:, :, i);
        end
        h_update = deparameterize(paramH_update);
        deparamH_update = reshape(h_update, 3, 3);
        deparamH_update = deparamH_update';
        proj1_update = zeros(2, num_point);
        for i = 1 : num_point
            pnt = deparameterize(pnt_scene_param_update(:, i));
            pnt = pnt / pnt(3);
            proj1_update(:, i) = pnt(1 : 2);
        end
        
        % compute new error
        error1_update = inlier1 - proj1_update;
        proj2_inhomo_update = zeros(3, num_point);
        for i = 1 : num_point
            proj2_update(:, i) = deparamH_update * ... 
                                 deparameterize(pnt_scene_param_update(:, i));
            proj2_inhomo_update(:, i) = proj2_update(:, i) / proj2_update(3, i);
        end
        proj2_inhomo_update = proj2_inhomo_update(1 : 2, :);
        error2_update = inlier2 - proj2_inhomo_update;
        cost_update = norm(error2_update(:))^2 + norm(error1_update(:))^2;

        % jump the loop
        if(cost_update > cost)
            lam = 10 * lam;
        else
            disp(cost_update);
            if((cost - cost_update) / cost < 1e-8)
                break;
            end
            n = n + 1;
            lam = lam / 10;
            paramH = paramH_update;
            H = deparamH_update;
            error1 = error1_update;
            error2 = error2_update;
            cost = cost_update;
            pnt_scene_param = pnt_scene_param_update;  
            proj2 = proj2_update;
        end
    end
    disp('number of iterations:');
    disp(n);
    H_LM = H;
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

% calculate Sampson error
function point = sampsonCorrection(H, point1, point2)
    x_hat = [point1(1 : 2), point2(1 : 2)]';
    point1 = point1';
    point2 = point2';
    % compute epsilon (Ah) and J
    epsilon = [-point1' * H(2, :)' + point2(2) * point1' * H(3, :)'; ...
               point1' * H(1, :)' - point2(1) * point1' * H(3, :)'];
    J = [-H(2, 1) + point2(2) * H(3, 1), -H(2, 2) + point2(2) * H(3, 2), ...
           0, point1(1) * H(3, 1) + point1(2) * H(3, 2) + H(3, 3); ...
           H(1, 1) - point2(1) * H(3, 1), H(1, 2) - point2(1) * H(3, 2), ...
           -(point1(1) * H(3, 1) + point1(2) * H(3, 2) + H(3, 3)), 0];
    % compute sampson error
    lam = (J * J') \ (-epsilon);
    delta = J' * lam;
    x_hat = x_hat + delta;
    point = x_hat(1 : 2)';
end

% compute jocabian matrix
function [A, B1, B2] = calJacobian(num_point, proj2, pnt_scene_param, H, paramH)
    A = zeros(2, 8, num_point);
    B1 = zeros(2, 2, num_point);
    B2 = zeros(2, 2, num_point);
    for i = 1:num_point
        pnt = proj2(:, i);
        pnt_scene = deparameterize(pnt_scene_param(:, i));

        % compute A
        A1_i = [1 / pnt(3), 0, -pnt(1) / pnt(3)^2;
                0, 1 / pnt(3), -pnt(2) / pnt(3)^2];
        A2_i = zeros(2,9);
        A2_i(1,1:3) = pnt_scene';
        A2_i(2,4:6) = pnt_scene';
        A2_i(3,7:9) = pnt_scene';

        part_H = zeros(9, 8);
        part_H(1, :) = -0.25 * (sinc(norm(norm(paramH) / 2))) * paramH';


        part_H(2:9, :) = sinc(norm(paramH) / 2) * 0.5 * eye(8) + ...
                         (1 / (4 * norm(paramH))) * ...
                         (derivSinc(norm(paramH) / 2)) * paramH * paramH';

        A_i = A1_i * A2_i * part_H;
        A(:, :, i) = A_i;

        % Compute B1
        B1_1_i = [1 / pnt_scene(3), 0, -pnt_scene(1) / pnt_scene(3)^2;
                  0, 1 / pnt_scene(3), -pnt_scene(2) / pnt_scene(3)^2];

        B1_2_i = zeros(3, 2);
        B1_2_i(1, :) = -0.25 * (sinc(norm(pnt_scene_param(:, i)) / 2)) * ...
                       pnt_scene_param(:, i)';
        B1_2_i(2 : 3, :) = sinc(norm(pnt_scene_param(:, i)) / 2) * 0.5 * ...
                           eye(2) + (1/(4 * norm(pnt_scene_param(:, i)))) * ...
                           (derivSinc(norm(pnt_scene_param(:, i)) / 2)) * ...
                           pnt_scene_param(:, i) * pnt_scene_param(:, i)';

        B1_i = B1_1_i * B1_2_i;
        B1(:, :, i) = B1_i;

        % compute B2
        B2_1_i = [1 / pnt(3), 0, -pnt(1) / pnt(3)^2;
                  0, 1 / pnt(3), -pnt(2) / pnt(3)^2];

        B2_2_i = H;

        B2_3_i = zeros(3,2);
        B2_3_i(1, :) = -0.25 * (sinc(norm(pnt_scene_param(:, i)) / 2)) * ...
                      pnt_scene_param(:, i)';
        B2_3_i(2 : 3, :) = sinc(norm(pnt_scene_param(:, i)) / 2) * 0.5 * ...
                           eye(2) + (1/(4 * norm(pnt_scene_param(:, i)))) * ...
                           (derivSinc(norm(pnt_scene_param(:, i)) / 2)) * ...
                           pnt_scene_param(:, i) * pnt_scene_param(:, i)';
        B2_i = B2_1_i * B2_2_i * B2_3_i;
        B2(:, :, i) = B2_i;
    end
end