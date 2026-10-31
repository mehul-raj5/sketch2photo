function h = fspecial(type, n)
% Minimal stand-in for the image package's fspecial, only the case FeatureSIM.m uses: fspecial('average', F).
if ~strcmp(type, 'average')
  error('fspecial shim: only ''average'' is implemented');
end
h = ones(n, n) / (n * n);
end
