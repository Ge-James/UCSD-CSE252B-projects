% feature detection
function [row, col] = featureDetection(image, w_size1, threshold, w_size2)
    % read the image in RGB format
    i = imread(image);

    % convert RGB to gray scale
    grayImage = rgb2gray(i);
    %grayImage = double(grayImage);

    % calculate gradient images
    K = [-1, 8, 0, -8, 1] / 12;
    Ix = imfilter(grayImage, K);
    Iy = imfilter(grayImage, K');

    % calculate Isquare and IxIy
    IxSquare = Ix .* Ix;
    IxIy = Ix .* Iy;
    IySquare = Iy .* Iy;

    % calculate minor eigenvalue image
    eigenImage = calEigenImage(IxSquare, IxIy, IySquare, w_size1);

    % set 0 if below threshold
    threshEigenImage = eigenImage .* (eigenImage >= threshold);

    % non maximum suppression
    % maximum filter
    Imax = ordfilt2(threshEigenImage, w_size2 * w_size2, ones(w_size2, w_size2));
    % compare two image, generate image J
    imageJ = threshEigenImage .* (threshEigenImage >= Imax);
    
    % find the coordinate of nonzero elements
    %[row, col] = find(imageJ);
    
    % find the coordinate of corner
    corner = findCorner(IxSquare, IxIy, IySquare, imageJ, w_size1);
    [row, col] = find(corner);
    
% calculate minor eigenvalue image
function m = calEigenImage(IxSquare, IxIy, IySquare, w_size)
    len = size(IxIy);
    m = zeros(len(1), len(2));
    for i = 1 : len(1)
        for j = 1 : len(2)
            [N, b] = calGradMatrix(IxSquare, IxIy, IySquare, i, j, w_size);
            m(i, j) = 0.5 * (trace(N) - sqrt(trace(N) ^ 2 - 4 * det(N)));
        end
    end

% Calculate gradient matrix
function [m, b] = calGradMatrix(IxSquare, IxIy, IySquare, i, j, w_size)
    m = zeros(2, 2);
    b = zeros(2, 1);
    half = (w_size - 1) / 2;
    len = size(IxIy);
    i_start = max(i - half, 1);
    j_start = max(j - half, 1);
    i_end = min(i + half, len(1));
    j_end = min(j + half, len(2));
    m(1, 1) = sum(sum(IxSquare(i_start : i_end, j_start : j_end)));
    m(1, 2) = sum(sum(IxIy(i_start : i_end, j_start : j_end)));
    m(2, 1) = m(1, 2);
    m(2, 2) = sum(sum(IySquare(i_start : i_end, j_start : j_end)));
    for p = i_start : i_end
        for q = j_start : j_end
            b(1) = b(1) + double(p) * double(IxSquare(p, q)) + double(q) * double(IxIy(p, q));
            b(2) = b(2) + double(q) * double(IySquare(p, q)) + double(p) * double(IxIy(p, q));
        end
    end

% find corner coordinates
function m = findCorner(IxSquare, IxIy, IySquare, imageJ, w_size)
    len = size(imageJ);
    m = zeros(len(1), len(2));
    for i = 1 : len(1)
        for j = 1 : len(2)
            if (imageJ(i, j) > 0)
                [N, b] = calGradMatrix(IxSquare, IxIy, IySquare, i, j, w_size);
                coord = N \ b;
                coord = coord';
                m(round(coord(1)), round(coord(2))) = 1;
            end
        end
    end

