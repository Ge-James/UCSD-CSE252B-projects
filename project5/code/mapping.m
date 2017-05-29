% map points to epipolar lines
function mapping(F, image1, image2)
    win_size = 20;
    color = ['r', 'y', 'g'];
    points = [200, 300, 1;
              300, 800, 1;
              420, 760, 1]';
    lines = [];
    for i = 1 : 3
        lines(:, i) = F * points(:, i);
    end
    
    % show the feature matching image
    figure(5)
    subplot(1, 2, 1);
    imshow(imread(image1));
    hold on;
    for i = 1 : 3
        plot(points(2, i), points(1, i), 's', 'MarkerSize', win_size, 'Color', color(i));
    end
    subplot(1, 2, 2);
    imshow(imread(image2));
    hold on;
    for i = 1 : 3
        syms x1 x2
        f = [x1, 0, 1] * lines(:, i);
        x1 = double(solve(f));
        f = [x2, 1024, 1] * lines(:, i);
        x2 = double(solve(f));
        line([0, 1024], [x1, x2], 'color', color(i));
        hold on;
    end
end