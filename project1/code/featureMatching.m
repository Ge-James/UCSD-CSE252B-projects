% feature matching
function match = featureMatching(image1, image2, row1, col1, row2, col2, w_size, simThresh, ratioThresh)
    I1 = imread(image1);
    I2 = imread(image2);
    Image1 = rgb2gray(I1);
    Image2 = rgb2gray(I2);
    length1 = size(row1);
    len1 = length1(1);
    length2 = size(row2);
    len2 = length2(1);
    match = zeros(len1, len2);
    % calculate correlation coefficient matrix
    correl = zeros(len1, len2);
    for i = 1 : len1
        [win1, size1] = fetchWindow(Image1, size(Image1), row1, col1, i, w_size);
        for j = 1 : len2
            [win2, size2] = fetchWindow(Image2, size(Image2), row2, col2, j, w_size);
            if size1 == w_size^2 && size2 == w_size^2
                correl(i, j) = corr2(win1, win2);
            else
                correl(i,j) = -1.0;
            end
        end
    end
    % one to one match
    maxValue = max(max(correl));
    count = 0;
    while maxValue > simThresh
        [X, Y] = find(correl == maxValue);
        x = X(1);
        y = Y(1);
        correl(x, y) = -1;
        nextMaxValue = max(max(correl(x,:)), max(correl(:,y)));
        if (1.0 - maxValue) < (1.0 - nextMaxValue) * ratioThresh
            count = count + 1;
            % constrain to the proximity
            distance = sqrt((row1(x) - row2(y))^2 + (col1(x) - col2(y))^2);
            if (distance < 150 && distance > 50 && col1(x) > col2(y) && abs(row1(x) - row2(y)) < 30)
                match(x, y) = 1;
            end
        end;
        correl(x,:) = -1;
        correl(:,y) = -1;
        maxValue = max(max(correl));
    end
    disp(count);

% fetch the window
function [win, size] = fetchWindow(image, len, row, col, i, w_size)
    half = (w_size - 1) / 2;
    i_start = max(row(i) - half, 1);
    j_start = max(col(i) - half, 1);
    i_end = min(row(i) + half, len(1));
    j_end = min(col(i) + half, len(2));
    win = image(i_start : i_end, j_start : j_end);
    size = (i_end - i_start + 1) * (j_end - j_start + 1);
            
            
    