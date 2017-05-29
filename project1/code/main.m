% This function is going to detect the % 
% corner of two images and match them  %

% input image
image1 = 'price_center20.JPG';
image2 = 'price_center21.JPG';

% set parameter
w_size1 = 9;
threshold = 60;
w_size2 = 11;
simThresh = 0.5;
ratioThresh = 0.825;

% corner detection
[row1, col1] = featureDetection(image1, w_size1, threshold, w_size2);
[row2, col2] = featureDetection(image2, w_size1, threshold, w_size2);

% count # features
x1 = row1(:);
x2 = row2(:);
y1 = col1(:);
y2 = col2(:);
disp(size(x1));
disp(size(x2));

% show the feature image
subplot(1, 2, 1);
imshow(rgb2gray(imread(image1)));
hold on;
scatter(y1, x1, w_size1 * w_size1, 's');
subplot(1, 2, 2)
imshow(rgb2gray(imread(image2)));
hold on;
scatter(y2, x2, w_size1 * w_size1, 's');

% feature matching
match = featureMatching(image1, image2, row1, col1, row2, col2, w_size2, simThresh, ratioThresh);
disp(sum(sum(match)));

% show the feature matching image
% subplot(1, 2, 1);
% imshow(rgb2gray(imread(image1)));
% hold on;
% length1 = size(row1);
% length2 = size(row2);
% for i = 1 : length1(1)
%     for j = 1 : length2(1)
%         if match(i, j) == 1
%             plot([col1(i), col2(j)], [row1(i), row2(j)], '-');
%             scatter(col1(i), row1(i), w_size1 * w_size1, 's');
%         end
%     end
% end
% subplot(1, 2, 2);
% imshow(rgb2gray(imread(image2)));
% hold on;
% length1 = size(row1);
% length2 = size(row2);
% for i = 1 : length1(1)
%     for j = 1 : length2(1)
%         if match(i, j) == 1
%             plot([col1(i), col2(j)], [row1(i), row2(j)], '-');
%             scatter(col2(j), row2(j), w_size1 * w_size1, 's');
%         end
%     end
% end
    