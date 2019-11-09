%return matrix with normalized rows
function Rnorm = normalize_rows( R )
row_norms = sqrt(sum(R.^2,2));
[~,m] = size(R);
Rnorm = R./repmat(row_norms,1,m);
end

