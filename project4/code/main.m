% This function is going to detect the % 
% corner of two images and match them  %

clc, clear, close all;
format longg;

% input image
image1 = 'price_center20.JPG';
image2 = 'price_center21.JPG';

figure(1)
subplot(1, 2, 1);
imshow(rgb2gray(imread(image1)));
subplot(1, 2, 2)
imshow(rgb2gray(imread(image2)));


%-------- Question (a) -----------%
% feature detection %

% set parameter
w_size1 = 9;
threshold = 380;
w_size2 = 9;
w_sizeb = 21;
simThresh = 0.6;
ratioThresh = 0.7;

% corner detection
[row1, col1] = featureDetection(image1, w_size1, threshold, w_size2);
[row2, col2] = featureDetection(image2, w_size1, threshold, w_size2);

% count # features
x1 = row1(:);
x2 = row2(:);
y1 = col1(:);
y2 = col2(:);
disp('Question (a):');
disp('number of features in figure 1:');
disp(size(x1, 1));
disp('number of features in figure 2:');
disp(size(x2, 1));

% show the feature image
figure(2)
subplot(1, 2, 1);
imshow(rgb2gray(imread(image1)));
hold on;
scatter(y1, x1, w_size1 * w_size1, 's');
subplot(1, 2, 2)
imshow(rgb2gray(imread(image2)));
hold on;
scatter(y2, x2, w_size1 * w_size1, 's');



%-------- Question (b) -----------%
% feature matching %

% feature matching
match = featureMatching(image1, image2, row1, col1, row2, col2, w_sizeb, simThresh, ratioThresh);
disp('Question (b):');
disp('number of matchings:');
disp(sum(sum(match)));

% show the feature matching image
figure(3)
subplot(1, 2, 1);
imshow(rgb2gray(imread(image1)));
hold on;
length1 = size(row1);
length2 = size(row2);
for i = 1 : length1(1)
    for j = 1 : length2(1)
        if match(i, j) == 1
            plot([col1(i), col2(j)], [row1(i), row2(j)], '-');
            scatter(col1(i), row1(i), w_size1 * w_size1, 's');
        end
    end
end
subplot(1, 2, 2);
imshow(rgb2gray(imread(image2)));
hold on;
length1 = size(row1);
length2 = size(row2);
for i = 1 : length1(1)
    for j = 1 : length2(1)
        if match(i, j) == 1
            plot([col1(i), col2(j)], [row1(i), row2(j)], '-');
            scatter(col2(j), row2(j), w_size1 * w_size1, 's');
        end
    end
end


%---------- Question (c) ------------%
% outliers rejection %

% extract coordinates of matching in question (b)
point2DOrig1 = [];
point2DOrig2 = [];
for i = 1 : size(row1, 1)
    for j = 1 : size(row2, 1)
        if match(i, j) == 1
            point2DOrig1 = [point2DOrig1; [row1(i), col1(i)]];
            point2DOrig2 = [point2DOrig2; [row2(j), col2(j)]];
        end
    end
end

% MSAC method
[inlierIndex, trials] = MSAC(point2DOrig1, point2DOrig2);
inlier1 = [];
inlier2 = [];
for i = 1 : length(inlierIndex)
    if inlierIndex(i) == 1
        inlier1 = [inlier1; point2DOrig1(i, :)];
        inlier2 = [inlier2; point2DOrig2(i, :)];
    end
end
disp('Question (c):');
disp('number of inliers:');
disp(sum(inlierIndex));
disp('number of trials:');
disp(trials);

% show the feature matching image
figure(4)
subplot(1, 2, 1);
imshow(rgb2gray(imread(image1)));
hold on;
for i = 1 : length(inlierIndex)
    if inlierIndex(i) == 1
        plot([point2DOrig1(i, 2), point2DOrig2(i, 2)], ...
             [point2DOrig1(i, 1), point2DOrig2(i, 1)], '-');
        scatter(point2DOrig1(i, 2), point2DOrig1(i, 1), w_size1 * w_size1, 's');
    end
end
subplot(1, 2, 2);
imshow(rgb2gray(imread(image2)));
hold on;
for i = 1 : length(inlierIndex)
    if inlierIndex(i) == 1
        plot([point2DOrig2(i, 2), point2DOrig1(i, 2)], ...
             [point2DOrig2(i, 1), point2DOrig1(i, 1)], '-');
        scatter(point2DOrig2(i, 2), point2DOrig2(i, 1), w_size1 * w_size1, 's');
    end
end



%----------- Question (d) ------------%
% linear estimation %

[H_norm, T2, T1] = linearEstimation(inlier2, inlier1);
format longg;
H_norm = - H_norm;

% scale P with ||P||Fro = 1
H = T2 \ H_norm * T1;
H = H / norm(H, 'fro');
H_DLT = - H;
disp('Question (d):');
disp('H matrix DLT:');
disp(H_DLT);

uni = ones(size(inlier1, 1), 1);
xEst = -H_DLT * [inlier1, uni]';
paramW = xEst(3, :);
xEst = xEst ./ paramW;
% disp(xEst);



%------------ Question (e) ------------%
% nonlinear estimation %

H_init = -H_DLT;
H_LM = levenbergEst(H_init, inlier1, inlier2);
H_LM = H_LM / norm(H_LM, 'fro');
disp('Question (e):')
disp('H_LM:')
disp(H_LM)