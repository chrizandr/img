I = imread(char(strcat(name, ".jpg")));

hog_ = vl_hog(single(I), 8, 'verbose') ;
imhog = vl_hog('render', hog_, 'verbose') ;
imwrite(imhog, char(strcat(name, "_hog_uoctti.jpg"))) ;

hog_ = vl_hog(single(I), 8, 'verbose', 'variant', 'dalaltriggs') ;
imhog = vl_hog('render', hog_, 'verbose', 'variant', 'dalaltriggs') ;
imwrite(imhog, char(strcat(name + "_hog_dalaltriggs.jpg"))) ;

I2 = fliplr(I);
hog_ = vl_hog(single(I2), 8, 'verbose') ;
hogflip = fliplr(hog);
imHog = vl_hog('render', hog_) ;
imhogflip = vl_hog('render', hogflip) ;
imwrite(imHog, char(strcat(name, "_hog_flipped.jpg"))) ;
imwrite(imhogflip, char(strcat(name, "_hog_flipped2.jpg"))) ;

A = [3, 4, 5, 9, 21];
for i = 1:length(A)
       hog_ = vl_hog(single(I), 8, 'verbose', 'numOrientations', A(i)) ;
       imhog = vl_hog('render', hog_, 'verbose', 'numOrientations', A(i)) ;
       imwrite(imhog, char(strcat(name, "_orient", num2str(A(i)), ".jpg")));
end


