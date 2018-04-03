file_path = './data/makeup_with_labels/val/yes_makeup/';
img_path_list = dir(strcat(file_path,'*.jpg'));
img_num = length(img_path_list);
if img_num > 0
    for j = 1:img_num
        image_name = img_path_list(j).name;
        image = imread(strcat(file_path, image_name));
        % resize
        resize_img = imresize(image,[128 128]);
        
        % save img
         save_dir = './data/output/dev/yes_makeup/';
         save_name = fullfile(save_dir, img_path_list(j).name);
         imwrite(resize_img, save_name);
    end
end
