% Interpretable cnn for big five pesonality traits using audio data %
% Generate cam mapping on clip-spectrogram %

% Data loader.
camfolder = '.../path/to/load/cam/of/each/personality/trait';
spectofolder = '.../path/to/load/corresponding/input/spectrogram';

% Load cam data.
filePattern = fullfile(camfolder, '*.mat');
matFiles = dir(filePattern);
for k = 1:length(matFiles)
  baseFileName = matFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  camData(k) = load(fullFileName);
end

% Load clip spectrogram
filePattern1 = fullfile(spectofolder, '*.mat');
matFiles1 = dir(filePattern1);
for k = 1:length(matFiles1)
  baseFileName1 = matFiles1(k).name;
  fullFileName1 = fullfile(myFolder1, baseFileName1);
  fprintf(1, 'Now reading %s\n', fullFileName1);
  spectoData(k) = load(fullFileName1);
end

for i = 1:length(camData)                      
    cam_FileName = camData(i);    
    specto_FileName = spectoData(i); 
    camstruct = struct2cell(cam_FileName);
    spectostruct = struct2cell(specto_FileName);
    cam_matrix = vertcat(camstruct{:});
    clip_matrix = vertcat(spectostruct{:});
    cam_double = double(cam_matrix);
    clip_double = double(clip_matrix);
    cam_min = min(cam_matrix(:));
    cam_max = max(cam_matrix(:));
    % Heat map.
    clip_min = min(clip_double(:));
    clip_max = max(clip_double(:));
    range1 = [clip_min clip_max];
    heatmap_gray = mat2gray(clip_double,range1);
    heatmap_x = gray2ind(heatmap_gray, 256);
    input_clip = ind2rgb(heatmap_x, jet(256));
    cam_resize = imresize(cam_double,[96 64]);
    div = cam_max - cam_min ;
    cam = cam_resize - cam_min ;
    cam_binary = cam_resize / div;
    BW1 = imbinarize(cam_binary,0.8);
    maskedRGBImage = input_clip .* repmat(BW1, 1, 1, 3);
    subplot(5,4,i); imshow(maskedRGBImage);
end
