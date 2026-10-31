function fsim_pairs(list_file, out_file)
% DECP's compute_fsim.m loop, per image: imread -> rgb2gray if 3 channels -> double -> FeatureSIM(probe, gallery).
% list_file: CSV lines "probe_path,gallery_path"; out_file: CSV "probe_path,fsim".
fid = fopen(list_file, 'r');
pairs = textscan(fid, '%s %s', 'Delimiter', ',');
fclose(fid);
probes = pairs{1}; galleries = pairs{2};
out = fopen(out_file, 'w');
for i = 1:numel(probes)
  probe = imread(probes{i});
  if size(probe, 3) == 3
    probe = rgb2gray(probe);
  end
  probe = double(probe);
  gallery = imread(galleries{i});
  if size(gallery, 3) == 3
    gallery = rgb2gray(gallery);
  end
  gallery = double(gallery);
  v = FeatureSIM(probe, gallery);
  fprintf(out, '%s,%.10f\n', probes{i}, v);
end
fclose(out);
end
