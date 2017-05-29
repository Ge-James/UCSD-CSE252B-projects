% two-view optimal triangulation
function pnt3D = triangulate(F, inlier1, inlier2)
    num_pnt = size(inlier1, 1);
    % uni = ones(num_pnt, 1);
    % inlier1_homo = [inlier1, uni];
    % inlier2_homo = [inlier2, uni];
    inlier1_homo = inlier1;
    inlier2_homo = inlier2;
    pnt3D = zeros(num_pnt, 4);
    pnt2D1 = zeros(num_pnt, 3);
    pnt2D2 = zeros(num_pnt, 3);
    
    % compute matrix P
    W = [0, 1, 0;
         -1, 0, 0;
         0, 0, 0];
    Z = [0, -1, 0;
         1, 0, 0;
         0, 0, 1];
    P = [eye(3),zeros(3, 1)];
    [~, ~, V] = svd(P);
    pnt1 = V(:, end);
    [U, D, V] = svd(F);
    D_prime = D;
    D_prime(3, 3) = (D(1, 1) + D(2, 2)) / 2.0;
    m = U * Z * D_prime * V';
    e_prime = -U(:, 3);
    proj_prime = [m, e_prime];

    disp('doing triangulation, this may take a while.');
    % do triangulation for each point
    for i = 1 : num_pnt
        point1 = inlier1_homo(i, :);
        point2 = inlier2_homo(i, :);
        
        % compute matrix Fs
        T  = [point1(3), 0, -point1(1);
              0, point1(3), -point1(2);
              0, 0, point1(3)];
        T_prime = [point2(3), 0, -point2(1);
                   0, point2(3), -point2(2);
                   0, 0, point2(3)];
        Fs = inv(T_prime') * F * inv(T);
        
        % compute epipoles of Fs
        [~, ~, V] = svd(Fs);
        e = V(:, end);
        [~, ~, V] = svd(Fs');
        e_prime = V(:, end);
        % scale the epipoles
        e = e / sqrt(e(1)^2 + e(2)^2);
        e_prime = e_prime / sqrt(e_prime(1)^2 + e_prime(2)^2);

        % compute the rotation matrices
        R  = [e(1), e(2), 0;
              -e(2), e(1), 0;
              0, 0, 1];
        R_prime = [e_prime(1), e_prime(2), 0;
                   -e_prime(2), e_prime(1), 0;
                   0, 0, 1];
               
        % F matrix in special form
        Fs = R_prime * Fs * R';
        f = e(3);
        f_prime = e_prime(3);
        a = Fs(2,2);
        b = Fs(2,3);
        c = Fs(3,2);
        d = Fs(3,3);

        % solve g(t) for t
        syms t
        g_t = t * ((a * t + b)^2 + f_prime^2 * (c * t + d)^2)^2 - ...
              (a * d - b * c) * (1 + f^2 * t^2)^2 * (a * t + b) * ...
              (c * t + d);
        tmp_t = solve(g_t);
        t = double(vpa(tmp_t));
        cost = zeros(6, 1);
        for n = 1 : 6
            real_t = real(t(n));
            cost(n) = (real_t)^2 / (1 + f^2 * (real_t)^2) + ...
                      (c * real_t + d)^2 / ((a * real_t + b)^2 + ...
                      f_prime^2 * (c * real_t + d)^2);
        end
        % find t with minimum cost
        [~,index] = min(cost);
        t_opt = t(index);
        
        % compute x_hat
        line = [t_opt * f, 1, -t_opt]';
        line_prime = [-f_prime * (c * t_opt + d), a * t_opt + b, c * t_opt + d]';
        x_hat  = [-line(1) * line(3), -line(2) * line(3), line(1)^2 + line(2)^2]';
        x_hat_prime = [-line_prime(1) * line_prime(3), -line_prime(2) * ...
                       line_prime(3), line_prime(1)^2 + line_prime(2)^2]';

        % correct points mapped back to original coordinates
        tmp_pnt1 = (inv(T) * R' * x_hat)';
        pnt2D1(i, :) = tmp_pnt1 / tmp_pnt1(3);
        tmp_pnt2 = (inv(T_prime) * R_prime' * x_hat_prime)';
        pnt2D2(i, :) = tmp_pnt2 / tmp_pnt2(3);

        % compute the line
        line_prime = F * pnt2D1(i, :)';
        line_orth_prime = [-line_prime(2) * pnt2D2(i, 3), line_prime(1) * ...
                     pnt2D2(i, 3), line_prime(2) * pnt2D2(i, 1) - ...
                     line_prime(1) * pnt2D2(i,2)]';
        plane = proj_prime' * line_orth_prime;

        % compute the 3D point
        P_plus = P' * inv(P * P');
        pnt2 = P_plus * pnt2D1(i, :)';
        tmp_3D_pnt = [pnt2(1) * plane(4) * pnt1(4), pnt2(2) * plane(4) * pnt1(4), ...
                       pnt2(3) * plane(4) * pnt1(4), -pnt1(4) * (plane(1) * pnt2(1) + ...
                       plane(2) * pnt2(2) + plane(3) * pnt2(3))];
        pnt3D(i, :) = tmp_3D_pnt / tmp_3D_pnt(4);
    end
    disp('triangulation done.');
end