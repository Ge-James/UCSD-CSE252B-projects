% 3 poing algorithm of Finsterwalder
function [R, t, cnt] = finsterWalder(point2D, point3D, sampleIndex)
    p1 = point3D(sampleIndex(1), :);
    p2 = point3D(sampleIndex(2), :);
    p3 = point3D(sampleIndex(3), :);
    
    q1 = point2D(sampleIndex(1), :);
    q2 = point2D(sampleIndex(2), :);
    q3 = point2D(sampleIndex(3), :);
    
    j1 = q1 / norm(q1);
    j2 = q2 / norm(q2);
    j3 = q3 / norm(q3);

    cos_alpha = dot(j2,j3);
    cos_beta = dot(j1,j3);
    cos_gamma = dot(j1,j2);

    a = norm(p2 - p3);
    b = norm(p1 - p3);
    c = norm(p1 - p2);

    G = c^2 * (c^2 * (1 - cos_beta^2) - b^2 * (1 - cos_gamma^2));
    H = b^2 * (b^2 - a^2) * (1 - cos_gamma^2) + c^2 * (c^2 + 2 * a^2) * ...
        (1 - cos_beta^2) + 2 * b^2 * c^2 * (-1 + cos_alpha * cos_beta * cos_gamma);
    I = b^2 * (b^2 - c^2) * (1 - cos_alpha^2) + a^2 * (a^2 + 2 * c^2) * ...
        (1 - cos_beta^2) + 2 * a^2 * b^2 * (-1 + cos_alpha * cos_beta * cos_gamma);
    J = a^2 * (a^2 * (1 - cos_beta^2) - b^2 * (1-cos_alpha^2));

    % find a real root
    lambda = roots([G, H, I, J]);
    for i = 1 : length(lambda)
        if isreal(lambda(i))
            realLambda = lambda(i);
            break;
        end
    end

    A = 1 + realLambda;
    B = -cos_alpha;
    C = (b^2 - a^2) / b^2 - realLambda * (c^2 / b^2);
    D = -realLambda * cos_gamma;
    E = (a^2 / b^2 + realLambda * c^2 / b^2) * cos_beta;
    F = -a^2 / b^2 + realLambda * ((b^2 - c^2) / b^2);

    p = sqrt(B^2 - A * C);
    q = sign(B * E - C * D) * sqrt(E^2 - C * F);

    m = [-B + p; -B - p] / C;
    n = [-E + q; -E - q] / C;

    pos = 1;
    u = [];
    v = [];
    % find all real roots for u
    for i = 1 : 2 
        A1 = b^2 - m(i)^2 * c^2;
        B1 = 2 * (c^2 * (cos_beta-n(i)) * m(i) - b^2 * cos_gamma);
        C1 = -c^2 * n(i)^2 + 2 * c^2 * n(i) * cos_beta + b^2 - c^2;
        possi_u = roots([A1 B1 C1]);
        for h = 1:length(possi_u)
            if isreal(possi_u(h))
                u(pos) = possi_u(h);
                v(pos) = u(pos) .* m(i) + n(i);
                pos = pos + 1;
            end
        end
    end
    
    cnt = length(u);
    if cnt == 0
        R = [];
        t = [];
        return
    end

    s1 = zeros(1,length(u));
    s2 = zeros(1,length(u));
    s3 = zeros(1,length(u));
    for i = 1 : length(u)
        s1(i) = sqrt(a^2 / (u(i)^2 + v(i)^2 - 2 * u(i) * v(i) * cos_alpha));
        s2(i) = u(i) * s1(i);
        s3(i) = v(i) * s1(i);
    end

    p_1 = j1' * s1;
    p_2 = j2' * s2;
    p_3 = j3' * s3;
    
    % compute the camera pose R and t
    R = zeros(3, 3, size(p_1,2));
    t = zeros(3, 1, size(p_1,2));
    threePoint3D = [p1; p2; p3];
    for i = 1 :size(p_1, 2)
        [R(:, :, i), t(:, :, i)] = calCameraPose(threePoint3D(:, 1 : 3), [p_1(:, i), p_2(:, i), p_3(:, i)]');
    end
end
    
    
                
        
        
        
        
        
        
        
        