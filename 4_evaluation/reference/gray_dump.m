function gray_dump(list_file, out_dir)
% Writes rgb2gray(imread(p)) of every path in list_file (one per line) to out_dir/<name>.png.
fid = fopen(list_file, 'r');
paths = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
paths = paths{1};
for i = 1:numel(paths)
  im = imread(paths{i});
  if size(im, 3) == 3
    im = rgb2gray(im);
  end
  [~, name, ~] = fileparts(paths{i});
  imwrite(im, fullfile(out_dir, [name '.png']));
end
end
