clc;
clear all;
close all;
%% camera sensor parameters
camera = struct('ImageSize',[480 640],'PrincipalPoint',[320 240],...
                'FocalLength',[320 320],'Position',[1.8750 0 1.2000],...
                'PositionSim3d',[0.5700 0 1.2000],'Rotation',[0 0 0],...
                'LaneDetectionRanges',[6 30],'DetectionRanges',[6 50],...
                'MeasurementNoise',diag([6,1,1]));
focalLength    = camera.FocalLength;
principalPoint = camera.PrincipalPoint;
imageSize      = camera.ImageSize;
% mounting height in meters from the ground
height         = camera.Position(3);  
% pitch of the camera in degrees
pitch          = camera.Rotation(2);  
            
camIntrinsics = cameraIntrinsics(focalLength, principalPoint, imageSize);
sensor        = monoCamera(camIntrinsics, height, 'Pitch', pitch);

%% define area to transform
distAheadOfSensor = 30; % in meters, as previously specified in monoCamera height input
spaceToOneSide    = 8;  % all other distance quantities are also in meters
bottomOffset      = 6;
outView   = [bottomOffset, distAheadOfSensor, -spaceToOneSide, spaceToOneSide]; % [xmin, xmax, ymin, ymax]
outImageSize = [NaN, 250]; % output image width in pixels; height is chosen automatically to preserve units per pixel ratio

birdsEyeConfig = birdsEyeView(sensor, outView, outImageSize);

videoReader = VideoReader('driftLeft.mp4');

f1 = figure;
f2 = figure;
f3 = figure;

frameNumber = 0;

%% process video frame by frame
while hasFrame(videoReader)

    frameNumber = frameNumber + 1;
    
    frame = readFrame(videoReader); % get the next video frame
    
    birdsEyeImage = transformImage(birdsEyeConfig, frame);
    birdsEyeImage = rgb2gray(birdsEyeImage);
    
    [h,w] = size(birdsEyeImage);

    figure(f1);
    imshow(birdsEyeImage);
    title("Frame n." + frameNumber );

    binImage = binarization(birdsEyeImage);

    figure(f2);
    imshow(binImage);
    title("Frame n." + frameNumber );

    pixelCount = sum(binImage);

    figure(f3); 
    plot(pixelCount)
    title("Frame n." + frameNumber );
    grid on; 
    grid minor;

    % 19 pixel di distanza tra centro dell'immagine e linea di corsia
    % corrispondono a circa 1,2 m di cui 1 m corrisponde a mezza larghezza della
    % macchina e 20 cm sono di margine (macchina stimata in 2 m e corsia 3,50 m)
    % la metà dell'immagine corrisponde al 125 esimo e il 126 esmio pixel

    % verifica che nei pixel tra il 106 e il 145 non ci siano troppi pixel
    % bianchi (più di 40)

    driftControl(pixelCount);


end