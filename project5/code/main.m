clc, clear, close all;
format longg;

% input image
image1 = 'IMG_5030.JPG';
image2 = 'IMG_5031.JPG';

% show the original picture
figure(1)
subplot(1, 2, 1);
imshow(imread(image1));
subplot(1, 2, 2)
imshow(imread(image2));



%-------- Question (a) -----------%
% feature detection %

% set parameter
w_size1 = 7;
threshold = 1000;
w_size2 = 7;
w_sizeb = 27;
simThresh = 0.5;
ratioThresh = 0.78;

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
imshow(imread(image1));
hold on;
scatter(y1, x1, w_size1 * w_size1, 's');
subplot(1, 2, 2)
imshow(imread(image2));
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
imshow(imread(image1));
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
imshow(imread(image2));
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



%---------- Question (c) ------------%
% outliers rejection %

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
imshow(imread(image1));
hold on;
for i = 1 : length(inlierIndex)
    if inlierIndex(i) == 1
        plot([point2DOrig1(i, 2), point2DOrig2(i, 2)], ...
             [point2DOrig1(i, 1), point2DOrig2(i, 1)], '-');
        scatter(point2DOrig1(i, 2), point2DOrig1(i, 1), w_size1 * w_size1, 's');
    end
end
subplot(1, 2, 2);
imshow(imread(image2));
hold on;
for i = 1 : length(inlierIndex)
    if inlierIndex(i) == 1
        plot([point2DOrig2(i, 2), point2DOrig1(i, 2)], ...
             [point2DOrig2(i, 1), point2DOrig1(i, 1)], '-');
        scatter(point2DOrig2(i, 2), point2DOrig2(i, 1), w_size1 * w_size1, 's');
    end
end



%---------- Question (d) ------------%
% linear estimation %

F_DLT = linearEstimation(inlier1, inlier2);
num_pnt = size(inlier1, 1);
format longg;
disp('Question (d):');
disp('F_DLT = ');
disp(F_DLT);

%{
for i = 1 : num_pnt
    tmp_pnt1 = inlier1(i, :);
    pnt1 = [tmp_pnt1, 1];
    tmp_pnt2 = inlier2(i, :);
    pnt2 = [tmp_pnt2, 1];
    disp(pnt2 * F_DLT * pnt1');
end
%}



%---------- Question (e) ------------%
% nonlinear estimation %

disp('Question (e):');
F = F_DLT;
uni = ones(num_pnt, 1);
inlier1 = [inlier1, uni];
inlier2 = [inlier2, uni];

pnt3D = triangulate(F, inlier1, inlier2);

[F_LM, cost_lst] = levenbergEst(F, inlier1, inlier2, pnt3D);
disp('cost for each iteration: ');
for i = 1 : size(cost_lst, 2)
    disp(cost_lst(i));
end
disp('F_LM = ');
disp(F_LM);



%------------- Question (f) --------------%
% point to line mapping %

F = F_LM;
mapping(F, image1, image2);
disp('Question (f):');
disp('The figure will show');