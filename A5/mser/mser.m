I = imread(char(strcat(name, ".jpg"))) ;

I = uint8(rgb2gray(I)) ;

[r,f] = vl_mser(I,'MinDiversity',0.7,...
                'MaxVariation',0.2,...
                'Delta',10) ;
f = vl_ertr(f) ;
vl_plotframe(f) ;
saveas(gcf, char(strcat(name, "_rframe.jpg")));

M = zeros(size(I)) ;
for x=r'
 s = vl_erfill(I,x) ;
 M(s) = M(s) + 1;
end

figure(2) ;
clf ; imagesc(I) ; hold on ; axis equal off; colormap gray ;
[c,h]=contour(M,(0:max(M(:)))+.5) ;
set(h,'color','y','linewidth',3) ;
saveas(gcf, char(strcat(name, "_mser.jpg")));

d = [32, 64, 96, 128, 160] ;

for i = 1:length(d)
    [r,f] = vl_mser(I,'MinDiversity',0.7,...
                'MaxVariation',0.2,...
                'Delta', d(i)) ;
    M = zeros(size(I)) ;
    for x=r'
     s = vl_erfill(I,x) ;
     M(s) = M(s) + 1;
    end

    figure(2) ;
    clf ; imagesc(I) ; hold on ; axis equal off; colormap gray ;
    [c,h]=contour(M,(0:max(M(:)))+.5) ;
    set(h,'color','y','linewidth',3) ;
    saveas(gcf, char(strcat(name, "_mser_delta", num2str(d(i)), ".jpg")));
end


[r,f] = vl_mser(I,'MinDiversity',0.7,...
                'MaxVariation',0.2,...
                'Delta',10,...
                'BrightOnDark',1,'DarkOnBright',0) ;
M = zeros(size(I)) ;
for x=r'
 s = vl_erfill(I,x) ;
 M(s) = M(s) + 1;
end

figure(2) ;
clf ; imagesc(I) ; hold on ; axis equal off; colormap gray ;
[c,h]=contour(M,(0:max(M(:)))+.5) ;
set(h,'color','y','linewidth',3) ;
saveas(gcf, char(strcat(name, "_mser_brondr.jpg")));