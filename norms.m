function  row_norm=norms(A,temp,n)
    coder.inline('always')
    row_norm= sqrt(sum(A.^2,n));
end
