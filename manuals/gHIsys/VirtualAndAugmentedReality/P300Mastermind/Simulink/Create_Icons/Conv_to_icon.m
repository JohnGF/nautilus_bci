function varargout = Conv_to_icon(imname,sz,varargin)

% -----------------------------------------------------------------
% function Conv_to_icon(Bild,sz);
% -----------------------------------------------------------------
% Input parameters: 
% -----------------
% imname:   name of the image file with suffix 
%           (must be stored within the current folder)
% sz:       image size in pixel (sz x sz)
% varargin: 'inv' generates an inverted binary image as output 
%           'jpg' generates a jpeg image as output
%           'pnt' generates a png image as output
%
% Output files:
% ------------
% depending on varargin, two files are stored to MATLABs current
% folder: 
% - if varargin is left empty, a binary  black-white-version of the 
%   input image is stored as bitmap, using the file name
%   imname_sz(1)xsz(2)_bin.bmp. 
%   Additionally, a text version of the image is stored as
%   imname_sz_bin-hex.txt
% - if varargin is set to 'inv', the input image is inverted.
%   A black-white-version of this image is stored as bitmap,
%   using the file name imname_sz(1)xsz(2)_bin_inv.bmp.
%   Additionally, a text version of the inverted image is stored
%   as imname_sz(1)xsz(2)_bin_hex_inv.txt.
% - if varargin is set to 'jpg', the input image is converted into a jpeg
%   compressed image using the file name imname_sz(1)xsz(2)_rgb.jpg o
%   Additionally, a text version of the jpeg image is stored
%   as imname_sz(1)xsz(2)_rgb.txt
% - if varargin is set to 'png', the input image is converted into a png
%   compressed image using the file name imname_sz(1)xsz(2)_rgb.png
%   Additionally, a text version of the jpeg image is stored
%   as imname_sz(1)xsz(2)_rgb.txt
% -----------------------------------------------------------------

%%
% read image
    [p,n,ex] = fileparts(imname);
    if isempty(ex)
        return;
    end
    fmt = imformats(ex(2:end));
    if isempty(fmt)
        return;
    end
    info = fmt.info(imname);
    if nargin < 3
        varargin{1} = '';
    end
    if strcmpi(info.ColorType,'indexed')
        [image,map,tran] = fmt.read(imname);
        image = ind2rgb(image,map);
    end
    if length(sz) < 2
        sz(2) = sz(1);
    end
    if info.Width ~= sz(2) || info.Height ~= sz(1)
        if ~exist('image','var')
            [image,~,tran] = fmt.read(imname);
        end
        image = imresize(image,sz);
        if ~isempty(tran)
            tran = imresize(tran,sz);
        end
    end
    
    switch varargin{1}
        case { 'jpg', 'png' }
            outfmt = imformats(varargin{1});   
            if strcmpi(info.ColorType,'grayscale')
                if ~exist('image','var')
                    [image,~,tran] = fmt.read(imname);
                end
                image = repmat(image,[1,1,3]);
            end
            if isempty(intersect(outfmt.ext,fmt.ext))
                if ~exist('image','var')
                    [image,~,tran] = fmt.read(imname);
                end
            end
            if exist('image','var')
                imname = fullfile(p,sprintf('%s_%dx%d_rgb.%s',n,sz,outfmt.ext{1}));
                if ~isempty(tran) && strcmp(varargin{1},'png');
                    outfmt.write(image,[],imname,'Alpha',tran);
                else
                    tran = [];
                    outfmt.write(image,[],imname);
                end
            end
            info = imfinfo(imname);

            switch ( info.ColorType )
                case 'grayscale'
                    color = 'grey';
                case 'truecolor'
                    if ~isempty(tran)
                        color = 'RGBA';
                    else
                        color = 'RGB';
                    end 
            end
            switch outfmt.ext{1}
                case 'jpg'
                    compress = 'jpeg';
                case 'png'
                    compress = 'png';
            end
            fid = fopen(imname,'rb');
            if fid < 0
                return;
            end
            content = fread(fid,inf,'*uint8')';
            fclose(fid);
            content = base64encode_f(content,3);
            content = sprintf('<Icon ID="%s" Format="%s" Encoding="base64" Compressed="%s">\n\t%s\n</Icon>',regexprep(n,'[^-+A-Za-z0-9_]','_'),color,compress,content(1:end-2));
        otherwise
            outfmt = imformats('bmp');
            if ~exist('image','var')
                image = fmt.read(imname);
            end
            if ~strcmpi(info.ColorType,'grayscale')
                image = rgb2gray(image);
            end
            
            switch(class(image))
                case 'double'
                    split = 0.5;
                case 'single'
                    split = 0.5;
                otherwise
                    split = ( intmax(class(image)) - intmin(class(image)) ) / 2;
            end
            nim = ones(size(image));
            nim(image < split) = 0;
            image = nim;
            clear nim;
            varargin{1} = lower(varargin{1});
            if strcmp(varargin{1},'inv')
                image = double(image == 0);
            else
                varargin{1} = 'bin';
            end
            imname = fullfile(p,sprintf('%s_%dx%d_%s.%s',n,sz,varargin{1},outfmt.ext{1}));
            outfmt.write(image,[],imname);
            content=sprintf('<Icon ID="%s" Format="binary" Encoding="hex">\n\t%s\n</Icon>',regexprep(n,'[^-+A-Za-z0-9_]','_'),dec2hex(bin2dec(num2str(reshape(image, 4, sz(1)*sz(2)/4)')))');            
    end
    fid=fopen(regexprep(imname,['\.' outfmt.ext{1} '$'],'.txt','once'),'wb');
    if fid >= 0
        fwrite(fid,content);
        fclose(fid);
    end
    if nargout > 0
        varargout{1} = content;
    end
    