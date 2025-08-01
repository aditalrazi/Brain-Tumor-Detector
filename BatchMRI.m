% Specify the path to the folder containing images and masks
clc;
clear;
close all;
warning off;

folderPath_images = 'C:\Users\Adit Al Razi\OneDrive - BUET\Desktop\Matlab\tumor_img';
folderPath_masks = 'C:\Users\Adit Al Razi\OneDrive - BUET\Desktop\Matlab\mask_img';

% Create output folder for saving results
outputFolder = 'C:\Users\Adit Al Razi\OneDrive - BUET\Desktop\Matlab\output_results';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Get a list of all image files in the folder
imageFiles = dir(fullfile(folderPath_images, '*.tif'));
maskFiles = dir(fullfile(folderPath_masks, '*_mask.tif'));

% Check if the number of image files and mask files match
if numel(imageFiles) ~= numel(maskFiles)
    error('Number of image files and mask files do not match.');
end

% Loop through each image file and process it
numFiles = numel(imageFiles);
dice = zeros(1, numFiles);
IoU = zeros(1, numFiles);
f1Score = zeros(1, numFiles);

for i = 1:numFiles
    % Construct the full file paths
    imagePath = fullfile(folderPath_images, imageFiles(i).name);
    maskPath = fullfile(folderPath_masks, maskFiles(i).name);
    
    if contains(maskFiles(i).name,'_mask')
        % ===== BRAIN TUMOR DETECTION AND EVALUATION CODE =====
        % (Previously in brainTwoDetectFunc_two function)
        
        % Load images
        I = imread(imagePath);
        IGndTr = imread(maskPath);
        Im = I; % Store original image for overlay
        
        % Converting to grayscale
        R = I(:,:,1);
        G = I(:,:,2);
        B = I(:,:,3);
        I = uint8(0.2989*double(R) + 0.5870*double(G) + 0.1140*double(B));
        
        gsAdj = I;
        
        % Skull stripping
        imbw = gsAdj > 20;
        imf = imfill(imbw,'holes');
        r = 20;
        se = strel("disk",r);
        erode_bw = imerode(imf,se);
        gsAdj = immultiply(gsAdj,erode_bw);
        
        I = gsAdj;
        
        % Applying high pass filter for noise removal
        N = 30;
        f = [0 0.1 0.15 1];
        m = [0 0 1 1];
        b_eq = firpm(N, f, m);
        hpf2d = b_eq.' * b_eq;
        Id = zeros(size(hpf2d));
        mid = (size(hpf2d, 1) + 1) / 2;
        Id(mid, mid) = 1;
        alpha = 1;
        H = Id + alpha * hpf2d;
        sharpened = imfilter(gsAdj, H, 'replicate');
        
        % Apply median filter to enhance the quality of image
        Median = medfilt2(sharpened); % 3x3 mean of pixels
        
        % Threshold segmentation
        level = multithresh(Median, 3); % Calculate multiple thresholds for the image
        seg_I = imquantize(Median, level); % Quantize the intensity values of the image
        RGB = label2rgb(seg_I); % Convert segmented image into a color image
        Threshold = rgb2gray(RGB); % Convert color image to grayscale
        
        im = Threshold; % Copy thresholded image
        im(im > 26 & im < 76) = 255; % Thresholding: set pixels equal to threshold to 255
        im(im > 76) = 0; % Thresholding: set pixels not equal to threshold to 0
        im(im < 26) = 0;
        im(im == 76) = 225;
        BW = im;
        
        % Watershed segmentation
        C = ~BW;
        D = -bwdist(C);
        L = watershed(D);
        Wi = label2rgb(C,'gray','w');
        lvl2 = graythresh(Wi);
        BW2 = im2bw(Wi,lvl2);
        
        BW2 = BW;
        
        % Morphological operations with robust error handling
        sout = BW2;
        label_img = bwlabel(sout);
        stats = regionprops(logical(sout),'Solidity','Area','BoundingBox');
        
        % Handle empty stats case
        if isempty(stats)
            no_tumor = 1;
            tumor = zeros(size(BW2));
            max_area = 0;
        else
            density = [stats.Solidity];
            area = [stats.Area];
            
            % Handle empty density/area arrays
            if isempty(density) || isempty(area)
                no_tumor = 1;
                tumor = zeros(size(BW2));
                max_area = 0;
            else
                high_dense_area = density > 0.2;
                
                % Check if any areas meet the density criteria
                if ~any(high_dense_area) || isempty(area(high_dense_area))
                    no_tumor = 1;
                    tumor = zeros(size(BW2));
                    max_area = 0;
                else
                    max_area = max(area(high_dense_area));
                    tumor_label = find(area == max_area);
                    tumor = ismember(label_img, tumor_label);
                    no_tumor = 0;
                    
                    if max_area > 500
                        % Tumor detected
                    else
                        % No tumor detected
                        no_tumor = 1;
                        tumor(tumor > 0) = 0;
                    end
                end
            end
        end
        
        BW3 = tumor;
        
        r = 100;
        se = strel("disk", r);
        tumor = imclose(BW3, se);
        
        % ===== SAVE OVERLAY RESULTS =====
        
        % Create overlays similar to hudai.m
        OLtumor = tumor;
        [M, N] = size(OLtumor);
        A = ones(M, N);
        A(OLtumor == 1) = 0;

        OLmask = IGndTr;
        A2 = ones(M, N);
        A2(OLmask == 1) = 0;

        % Create figure for saving (invisible to avoid popup)
        fig = figure('Visible', 'off');
        set(fig, 'Position', [100, 100, 1200, 400]); % Set figure size
        
        % Create subplot for detected tumor overlay
        subplot(1, 2, 1);
        h = imshow(Im);
        title(['MRI with Detected Tumor Overlay - Image ', num2str(i)]);
        hold on;
        set(h, 'AlphaData', A);
        hold off;

        % Create subplot for ground truth mask
        subplot(1, 2, 2);
        h1 = imshow(Im);
        title(['MRI with Ground Truth Mask - Image ', num2str(i)]);
        hold on;
        set(h1, 'AlphaData', A2);
        hold off;
        
        % Get base filename without extension
        [~, baseFileName, ~] = fileparts(imageFiles(i).name);
        
        % Save the combined overlay figure
        outputFileName = fullfile(outputFolder, sprintf('%s_overlay_results.png', baseFileName));
        saveas(fig, outputFileName);
        
        % Save individual images as well
        % Save detected tumor overlay
        fig2 = figure('Visible', 'off');
        h_det = imshow(Im);
        title(['Detected Tumor Overlay - ', baseFileName]);
        hold on;
        set(h_det, 'AlphaData', A);
        hold off;
        outputFileName_det = fullfile(outputFolder, sprintf('%s_detected_tumor.png', baseFileName));
        saveas(fig2, outputFileName_det);
        close(fig2);
        
        % Save ground truth overlay
        fig3 = figure('Visible', 'off');
        h_gt = imshow(Im);
        title(['Ground Truth Mask - ', baseFileName]);
        hold on;
        set(h_gt, 'AlphaData', A2);
        hold off;
        outputFileName_gt = fullfile(outputFolder, sprintf('%s_ground_truth.png', baseFileName));
        saveas(fig3, outputFileName_gt);
        close(fig3);
        
        % Close the main figure
        close(fig);
        
        % ===== RESULT EVALUATION =====
        
        % Determine the dice coefficient
        predictedImage = tumor;
        groundTruthImage = IGndTr;
        
        % Ensure the input images are binary
        predicted = logical(predictedImage);
        groundTruth = logical(groundTruthImage);
        
        % Calculate the Dice coefficient with robust error handling
        if no_tumor == 1
            intersection = nnz(predicted == groundTruth);
            dice(i) = 2 * intersection / 131072;
        else
            intersection = nnz(predicted & groundTruth);
            denominator = nnz(predicted) + nnz(groundTruth);
            if denominator == 0
                dice(i) = 0;
            else
                dice(i) = 2 * intersection / denominator;
            end
        end
        
        % Determine the IoU coefficient with robust error handling
        if no_tumor == 1
            union = 131072 - intersection;
            if union == 0
                IoU(i) = 0;
            else
                IoU(i) = intersection / union;
            end
        else
            union = nnz(predicted | groundTruth);
            if union == 0
                IoU(i) = 0;
            else
                IoU(i) = intersection / union;
            end
        end
        
        % Determine the F1 coefficient with robust error handling
        if no_tumor == 1
            truePositives = sum(groundTruth == predicted);
            falsePositives = sum(~groundTruth == predicted);
            falseNegatives = sum(groundTruth == ~predicted);
        else
            truePositives = sum(groundTruth & predicted);
            falsePositives = sum(~groundTruth & predicted);
            falseNegatives = sum(groundTruth & ~predicted);
        end
        
        % Handle division by zero cases
        if (truePositives + falsePositives) == 0
            precision = 0;
        else
            precision = truePositives / (truePositives + falsePositives);
        end
        
        if (truePositives + falseNegatives) == 0
            recall = 0;
        else
            recall = truePositives / (truePositives + falseNegatives);
        end
        
        if precision + recall == 0
            f1Score(i) = 0;
        else
            f1Score(i) = 2 * (precision * recall) / (precision + recall);
        end
        
        % Display progress
        fprintf('Processed image %d/%d: %s\n', i, numFiles, imageFiles(i).name);
        
        % ===== END OF BRAIN TUMOR DETECTION CODE =====
    end
end

% Calculate the average of the dice, IoU, and F1 score
diceValue = mean(dice);
IoUValue = mean(IoU);
f1ScoreValue = mean(f1Score);

% Standard deviation
diceStd = std(dice);
IoUStd = std(IoU);
f1ScrStd = std(f1Score);

% Calculate the normal distribution
x = linspace(0,1,1000);
pdfDice = normpdf(x,diceValue,diceStd);
pdfIoU = normpdf(x,IoUValue,IoUStd);
pdff1Scr = normpdf(x,f1ScoreValue,f1ScrStd);

% Create and save the distribution plot
fig_dist = figure('Name', 'Performance Distribution');
plot(x,pdfDice,'b', 'LineWidth', 2);
hold on;
plot(x,pdfIoU,'r', 'LineWidth', 2);
hold on;
plot(x,pdff1Scr,'g', 'LineWidth', 2);
legend('Dice coefficient','IoU Score','F1 Score');
title('Performance Metrics Distribution');
xlabel('Score');
ylabel('Probability Density');
grid on;

% Save the distribution plot
distributionFileName = fullfile(outputFolder, 'performance_distribution.png');
saveas(fig_dist, distributionFileName);

% Display the average scores
disp(['Average Dice Coefficient: ', num2str(diceValue),' and the standard deviation : ',num2str(diceStd)]);
disp(['Average IoU Score: ', num2str(IoUValue),' and the standard deviation : ',num2str(IoUStd)]);
disp(['Average F1 Score: ', num2str(f1ScoreValue),' and the standard deviation : ',num2str(f1ScrStd)]);

% Create a summary text file
summaryFileName = fullfile(outputFolder, 'results_summary.txt');
fid = fopen(summaryFileName, 'w');
fprintf(fid, 'Brain Tumor Detection Results Summary\n');
fprintf(fid, '=====================================\n\n');
fprintf(fid, 'Number of processed images: %d\n\n', numFiles);
fprintf(fid, 'Performance Metrics:\n');
fprintf(fid, '-------------------\n');
fprintf(fid, 'Average Dice Coefficient: %.4f ± %.4f\n', diceValue, diceStd);
fprintf(fid, 'Average IoU Score: %.4f ± %.4f\n', IoUValue, IoUStd);
fprintf(fid, 'Average F1 Score: %.4f ± %.4f\n', f1ScoreValue, f1ScrStd);
fprintf(fid, '\nIndividual Results:\n');
fprintf(fid, '------------------\n');
for i = 1:numFiles
    fprintf(fid, 'Image %d (%s): Dice=%.4f, IoU=%.4f, F1=%.4f\n', ...
        i, imageFiles(i).name, dice(i), IoU(i), f1Score(i));
end
fclose(fid);

fprintf('\nResults saved to: %s\n', outputFolder);
fprintf('- Individual overlay images saved for each processed image\n');
fprintf('- Combined overlay results saved for each image\n');
fprintf('- Performance distribution plot saved\n');
fprintf('- Results summary saved to text file\n');