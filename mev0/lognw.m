function result = lognw(x);
% lognw(x) is the same as log(x) but suppressing warnings about zero

if isempty(x),
   result = zeros(size(x));
else

   ind = find(x == 0);
   x(ind) = ones(size(ind));
   result = log(x);
   result(ind) = -Inf * ones(size(ind));

end
return;
