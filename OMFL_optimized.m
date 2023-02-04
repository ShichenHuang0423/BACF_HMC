% This function implements the BACF tracker.

function [results] = OMFL_optimized(params)

%   Setting parameters for local use.
search_area_scale   = params.search_area_scale;
output_sigma_factor = params.output_sigma_factor;
learning_rate       = params.learning_rate;
filter_max_area     = params.filter_max_area;
nScales             = params.number_of_scales;
scale_step          = params.scale_step;

features    = params.t_features;
video_path  = params.video_path;
s_frames    = params.s_frames;
pos         = floor(params.init_pos);
target_sz   = floor(params.wsize);
ground = params.ground_truth;
% ground_truth_bacf = params.ground_truth_bacf;
% ground_truth_kcf = params.ground_truth_kcf;
% ground_truth_ldes = params.ground_truth_ldes;
visualization  = params.visualization;
num_frames     = params.no_fram;
init_target_sz = target_sz;

%set the feature ratio to the feature-cell size
featureRatio = params.t_global.cell_size;
search_area = prod(init_target_sz / featureRatio * search_area_scale);
    
% when the number of cells are small, choose a smaller cell size
if isfield(params.t_global, 'cell_selection_thresh')
    if search_area < params.t_global.cell_selection_thresh * filter_max_area
        params.t_global.cell_size = min(featureRatio, max(1, ceil(sqrt(prod(init_target_sz * search_area_scale)/(params.t_global.cell_selection_thresh * filter_max_area)))));
        featureRatio = params.t_global.cell_size;
        search_area = prod(init_target_sz / featureRatio * search_area_scale);
    end  
end

global_feat_params = params.t_global;

if search_area > filter_max_area
    currentScaleFactor = sqrt(search_area / filter_max_area);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        sz = floor( base_target_sz * search_area_scale);                   % proportional area, same aspect ratio as the target
    case 'square'
        sz = repmat(sqrt(prod(base_target_sz * search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        sz = base_target_sz + sqrt(prod(base_target_sz * search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    otherwise
        error('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''');
end

% set the size to exactly match the cell size
sz = round(sz / featureRatio) * featureRatio;
% use_sz is the size of search area counting in cells
use_sz = floor(sz/featureRatio);


% construct the label function- correlation output, 2D gaussian function,
% with a peak located upon the target.
output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;
rg           = circshift(-floor((use_sz(1)-1)/2):ceil((use_sz(1)-1)/2), [0 -floor((use_sz(1)-1)/2)]);
cg           = circshift(-floor((use_sz(2)-1)/2):ceil((use_sz(2)-1)/2), [0 -floor((use_sz(2)-1)/2)]);
[rs, cs]     = ndgrid( rg,cg);
y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf           = fft2(y); %   FFT of y.

% construct cosine window
cos_window = single(hann(use_sz(1))*hann(use_sz(2))');

% Calculate feature dimension.
try
    im = imread([video_path '/img/' s_frames{1}]);
catch
    try
        im = imread(s_frames{1});
    catch
        %disp([video_path '/' s_frames{1}])
        im = imread([video_path '/' s_frames{1}]);
    end
end
if nScales > 0
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    scaleFactors = scale_step .^ scale_exp;
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

% initialize the projection matrix (x,y,h,w)
rect_position = zeros(num_frames, 4);
time = 0;

% allocate memory for multi-scale tracking
multires_pixel_template = zeros(sz(1), sz(2), size(im,3), nScales, 'uint8');

small_filter_sz = floor(base_target_sz/featureRatio);

loop_frame = 1;
for frame = 1:numel(s_frames)
    %load image
%     test = s_frames{frame}
    try
        im = imread([video_path '/img/' s_frames{frame}]);
    catch
        try
            im = imread([s_frames{frame}]);
        catch
            im = imread([video_path '/' s_frames{frame}]);
        end
    end
%     img = im;
    tic();
    
    %do not estimate translation and scaling on the first frame, since we
    %just want to initialize the tracker there
    if frame > 1
        for scale_ind = 1:nScales
            multires_pixel_template(:,:,:,scale_ind) = ...
                get_pixels(im, pos, round(sz*currentScaleFactor*scaleFactors(scale_ind)), sz);
        end

        xtf = fft2(bsxfun(@times,get_features(multires_pixel_template,features,global_feat_params,'fhog'),cos_window));
      
        
       
        % only use fhog features to estimate scale
        other_pixel_template(:,:,:) = ...
                get_pixels(im, pos, round(sz*currentScaleFactor*scaleFactors(scale_ind)), sz);
          
        xtf_gray = fft2(bsxfun(@times,get_features(other_pixel_template,features,global_feat_params,'gray'),cos_window));

       
        
        responsef = permute(sum(bsxfun(@times, conj(g_f{1}), xtf), 3), [1 2 4 3]);
        responsef_gray = permute(sum(bsxfun(@times, conj(g_f{3}), xtf_gray), 3), [1 2 4 3]);
        
        
        % use dynamic interp size
        interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
        
        responsef_padded = resizeDFT2(responsef, interp_sz);
        responsef_padded_gray = resizeDFT2(responsef_gray, interp_sz);
        
        % response in the spatial domain
        response = ifft2(responsef_padded, 'symmetric');
        response_gray = ifft2(responsef_padded_gray,'symmetric');   
         
        % find maximum peak
        [~, ~, sind] = ind2sub(size(response), find(response == max(response(:)), 1));
        response_fhog = response(:,:,sind);

        % use PSR to weight the features
        max_fhog = max(response_fhog(:));
        min_fhog = min(response_fhog(:));
        APCE_fhog=(max_fhog-min_fhog)^2/(mean(sum(abs(response-min_fhog))))*1000;

     
        if frame > 108
            ttt = 1;
        end
        max_gray = max(response_gray(:));
        min_gray = min(response_gray(:));
        APCE_gray=(max_gray-min_gray)^2/(mean(sum(abs(response-min_gray))))*1000;
        
        


        
        if numel(size(im)) > 2
			xtf_cn = fft2(bsxfun(@times,get_features(other_pixel_template,features,global_feat_params,'cn'),cos_window));
            xtf_saliency = fft2(bsxfun(@times,get_features(other_pixel_template,features,global_feat_params,'saliency'),cos_window));
            
         
            
            responsef_cn = permute(sum(bsxfun(@times, conj(g_f{2}), xtf_cn), 3), [1 2 4 3]);
            responsef_saliency = permute(sum(bsxfun(@times, conj(g_f{4}), xtf_saliency), 3), [1 2 4 3]);
            
            responsef_padded_cn = resizeDFT2(responsef_cn, interp_sz);
            responsef_padded_saliency = resizeDFT2(responsef_saliency, interp_sz);
            
            response_cn = ifft2(responsef_padded_cn,'symmetric');
            response_saliency = ifft2(responsef_padded_saliency,'symmetric');
               
               
            max_cn = max(response_cn(:));
            min_cn = min(response_cn(:));
            APCE_cn=(max_cn-min_cn)^2/(mean(sum(abs(response-min_cn))))*1000;
            
            max_saliency = max(response_saliency(:));
            min_saliency = min(response_saliency(:));
            APCE_saliency=(max_saliency-min_saliency)^2/(mean(sum(abs(response-min_saliency))))*1000;
            if APCE_cn<0.066
                k_cn=1; 
            end
            if APCE_fhog<0.32  
                k_fhog=1;
            end
            if APCE_gray<0.011
                k_gray=1;
            end
            

            wH = psr(response_fhog)*response_fhog;
            wG = psr(response_gray)*response_gray;
            wC = psr(response_cn)*response_cn;
            wS = psr(response_saliency)*response_saliency;
            
            
          
            response_fuse_FhogAndCn = wH .* wC;
            response_fuse_FhogAndGray = wH .* wG;
            response_fuse_FhogAndSaliency = wH .* wS;
            response_fuse_CnAndGray = wC .* wG;
            response_fuse_CnAndSaliency = wC .* wS;
            response_fuse_GrayAndSaliency = wG .* wS;
            % weighted by PSR
            wHC = psr(response_fuse_FhogAndCn)*response_fuse_FhogAndCn;
            wHG = psr(response_fuse_FhogAndGray)*response_fuse_FhogAndGray;
            wHS = psr(response_fuse_FhogAndSaliency)*response_fuse_FhogAndSaliency;
            wCG = psr(response_fuse_CnAndGray)*response_fuse_CnAndGray;
            wCS = psr(response_fuse_CnAndSaliency)*response_fuse_CnAndSaliency;
            wGS = psr(response_fuse_GrayAndSaliency)*response_fuse_GrayAndSaliency;
             % final fusion
            response_fuse = (wHC + wHG + wHS + wCG + wCS + wGS)/6;
%             response_fuse = response_fhog;
       

        else
            wH = psr(response_fhog)*response_fhog;
            wG = psr(response_gray)*response_gray;
            % fusion
            response_fuse_FhogAndGray = wH .* wG;
            % weighted by PSR
            wHG = psr(response_fuse_FhogAndGray)*response_fuse_FhogAndGray;
            % final fusion
            response_fuse = wHG;
%             response_fuse = response_fhog;
        end
          
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%         if frame == 2
%             csvwrite('response.csv',response_fuse*100);
%         end


%         figure(2);
%         A = imresize(ifftshift(response_fuse),4);
%         map = colormap(jet);
%         B = zeros(size(A,1),size(A,2),3);
%         for i=1:size(A,1)
%             for j=1:size(A,2)
%                 x = cell(A(i,j) * size(map,1));
%                 if x <= 0
%                     x = 1;
%                 end
%                 B(i,j,:) = map(x,:);
%             end
%         end
%         C = double(get_pixels(1:size(A,1),1:size(A,2),1:size(A,3),:)) / 255;
%         subplot(131);imshow(C)
%         subplot(132);imshow(B)
%         subplot(133);imshow(imadd(B*0.7,C*0.4),[])
        
        
%         resshift_fuse=fftshift(response_fuse);
%         i=num2str(frame);
%         handle_fuse=figure;
%         surf(resshift_fuse);
%         mesh(resshift_fuse);
% %             surfl(resshift_cn');
%         colormap(winter);
%         set(handle_fuse,'visible','off'); % 设置figure为不可视
%         xlabel('x(m)','FontName','Times New Roman','FontSize',15);
%         ylabel('zc(m) and y(m)','FontName','Times New Roman','FontSize',15);
%         zlabel('zc(m) and y(m)','FontName','Times New Roman','FontSize',15);
%         saveas(gcf,['G:\puser\多元视觉认知\融合特征图\试\','testw',w,'.jpg']);


%         if frame == 35
%             figure;
%             colormap('hot')
%             imagesc(response_fuse)
%             colorbar
%             
% %             f = figure(2);
% %             A = imresize(ifftshift(response_fuse),4);
% %             subplot(122);imshow(A);colormap(jet);
% %             subplot(121);
% %             B = patch(1:size(A,1),1:size(A,2),:);
% %             imshow(B);
% %             saveas(f,'heatmap.jpg','jpg');
%         end



%          if frame == 35
%              figure
%              depth2 = flipdim(response_cn,1);
%              h = pcolor(depth2);
%              axis equal
%              set(h,'edgecolor','none','facecolor','interp');
%              colorbar;
% %              subplot(1,2,2)
%              imshow(img)
%          end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        % find maximum peak on the final fused response map
        [row_fuse, col_fuse, ~ ] = ind2sub(size(response_fuse), find(response_fuse == max(response_fuse(:)), 1));
        disp_row_fuse = mod(row_fuse - 1 + floor((interp_sz(1)-1)/2), interp_sz(1)) - floor((interp_sz(1)-1)/2);
        disp_col_fuse = mod(col_fuse - 1 + floor((interp_sz(2)-1)/2), interp_sz(2)) - floor((interp_sz(2)-1)/2);
        
        % calculate translation
        translation_vec_fuse = round([disp_row_fuse, disp_col_fuse] * scaleFactors(sind));
        
        % set the scale
        currentScaleFactor = currentScaleFactor * scaleFactors(sind);
        % adjust to make sure we are not to large or to small
        if currentScaleFactor < min_scale_factor
            currentScaleFactor = min_scale_factor;
        elseif currentScaleFactor > max_scale_factor
            currentScaleFactor = max_scale_factor;
        end
        
        % update position
        old_pos = pos;
        pos = pos + translation_vec_fuse;
    end
    
    % extract training sample image region
    pixels = get_pixels(im,pos,round(sz*currentScaleFactor),sz);
    
    % extract features and do windowing
    xf = fft2(bsxfun(@times,get_features(pixels,features,global_feat_params,'fhog'),cos_window));
    xf_gray = fft2(bsxfun(@times,get_features(pixels,features,global_feat_params,'gray'),cos_window));
    
    if numel(size(im)) > 2
       xf_cn = fft2(bsxfun(@times,get_features(pixels,features,global_feat_params,'cn'),cos_window)); 
       xf_saliency = fft2(bsxfun(@times,get_features(pixels,features,global_feat_params,'saliency'),cos_window));
    else
       xf_cn = 0;
       xf_saliency = 0;
    end
    
         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       xf_k = bsxfun(@times,get_features(pixels,features,global_feat_params,'fhog'),cos_window);
%       [a_2,b_2,c_2] = size(xf_k);
%       xf_k=double(xf_k);
%       f_k=[];
%       for i=1:c_2
%         f_k=[f_k;xf_k(:,:,i)];
%       end
%       handle=figure;
%       mesh(f_k)
%      colormap winter
%      w=num2str(frame);
%      set(handle,'visible','off');
%      saveas(gcf,['G:\puser\多元视觉认知\融合特征图\xf\','testw',w,'.jpg']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    if (frame == 1)
        if numel(size(im)) > 2
           model_xf = {xf, xf_cn, xf_gray, xf_saliency};
        else
           model_xf = {xf, 0, xf_gray, 0};  
        end
        k_cn = 0;
        k_gray = 0;
        k_fhog = 0;
    else
        test_fhog = model_xf{1};
        test_cn = model_xf{2};
        test_gray = model_xf{3};
        model_xf = {((1 - learning_rate) * model_xf{1}) + (learning_rate * xf), ...
                    ((1 - learning_rate) * model_xf{2}) + (learning_rate * xf_cn), ...
                    ((1 - learning_rate) * model_xf{3}) + (learning_rate * xf_gray), ...
                    ((1 - learning_rate) * model_xf{4}) + (learning_rate * xf_saliency)};
        if k_cn == 1
            model_xf{2} = test_cn;
        end
        if k_gray == 1
            model_xf{3} = test_gray;
        end
        if k_fhog == 1
            model_xf{1}=test_fhog;
        end
        k_cn = 0;
        k_gray = 0;
        k_fhog = 0;
    end
   
            
     
    g_f = {single(zeros(size(xf))), single(zeros(size(xf_cn))), single(zeros(size(xf_gray))), single(zeros(size(xf_saliency)))};
    h_f = g_f;
    l_f = g_f;
    
    mu    = {1, 1, 1, 1};
    betha = {10,10,10,10};
    mumax = {10000,10000,10000,10000};
    i = 1;
    
    T = prod(use_sz);
    S_xx = {sum(conj(model_xf{1}) .* model_xf{1}, 3), sum(conj(model_xf{2}) .* model_xf{2}, 3), ...
            sum(conj(model_xf{3}) .* model_xf{3}, 3), sum(conj(model_xf{4}) .* model_xf{4}, 3)};
    params.admm_iterations = 2;
    
    %   ADMM
    while (i <= params.admm_iterations)
        %   solve for G- please refer to the paper for more details
        B = {S_xx{1} + (T * mu{1}), S_xx{2} + (T * mu{2}), S_xx{3} + (T * mu{3}), S_xx{4} + (T * mu{4})};
        S_lx = {sum(conj(model_xf{1}) .* l_f{1}, 3), sum(conj(model_xf{2}) .* l_f{2}, 3), sum(conj(model_xf{3}) .* l_f{3}, 3), sum(conj(model_xf{4}) .* l_f{4}, 3)};
        S_hx = {sum(conj(model_xf{1}) .* h_f{1}, 3), sum(conj(model_xf{2}) .* h_f{2}, 3), sum(conj(model_xf{3}) .* l_f{3}, 3), sum(conj(model_xf{4}) .* l_f{4}, 3)};
        g_f = {(((1/(T*mu{1})) * bsxfun(@times, yf, model_xf{1})) - ((1/mu{1}) * l_f{1})  + h_f{1}) - ...
            bsxfun(@rdivide,(((1/(T*mu{1})) * bsxfun(@times, model_xf{1}, (S_xx{1} .* yf))) ...
            - ((1/mu{1}) * bsxfun(@times, model_xf{1}, S_lx{1})) + (bsxfun(@times, model_xf{1}, S_hx{1}))), B{1}),...
            (((1/(T*mu{2})) * bsxfun(@times, yf, model_xf{2})) - ((1/mu{2}) * l_f{2})  + h_f{2}) - ...
            bsxfun(@rdivide,(((1/(T*mu{2})) * bsxfun(@times, model_xf{2}, (S_xx{2} .* yf))) ...
            - ((1/mu{2}) * bsxfun(@times, model_xf{2}, S_lx{2})) + (bsxfun(@times, model_xf{2}, S_hx{2}))), B{2}),...
            (((1/(T*mu{3})) * bsxfun(@times, yf, model_xf{3})) - ((1/mu{3}) * l_f{3})  + h_f{3}) - ...
            bsxfun(@rdivide,(((1/(T*mu{3})) * bsxfun(@times, model_xf{3}, (S_xx{3} .* yf))) ...
            - ((1/mu{3}) * bsxfun(@times, model_xf{3}, S_lx{3})) + (bsxfun(@times, model_xf{3}, S_hx{3}))), B{3}),...
            (((1/(T*mu{4})) * bsxfun(@times, yf, model_xf{4})) - ((1/mu{4}) * l_f{4})  + h_f{4}) - ...
            bsxfun(@rdivide,(((1/(T*mu{4})) * bsxfun(@times, model_xf{4}, (S_xx{4} .* yf))) ...
            - ((1/mu{4}) * bsxfun(@times, model_xf{4}, S_lx{4})) + (bsxfun(@times, model_xf{4}, S_hx{4}))), B{4})};
        
        %   solve for H
        h = {(T/((mu{1}*T)+ params.admm_lambda))  *   ifft2((mu{1}*g_f{1}) + l_f{1}),...
            (T/((mu{2}*T)+ params.admm_lambda))  *   ifft2((mu{2}*g_f{2}) + l_f{2}),...
            (T/((mu{3}*T)+ params.admm_lambda))  *   ifft2((mu{3}*g_f{3}) + l_f{3}),...
            (T/((mu{4}*T)+ params.admm_lambda))  *   ifft2((mu{4}*g_f{4}) + l_f{4})};
     
        [sx{1},sy{1},h{1}] = get_subwindow_no_window(h{1}, floor(use_sz/2) , small_filter_sz);
        [sx{2},sy{2},h{2}] = get_subwindow_no_window(h{2}, floor(use_sz/2) , small_filter_sz);
        [sx{3},sy{3},h{3}] = get_subwindow_no_window(h{3}, floor(use_sz/2) , small_filter_sz);
        [sx{4},sy{4},h{4}] = get_subwindow_no_window(h{4}, floor(use_sz/2) , small_filter_sz);
        t = {single(zeros(use_sz(1), use_sz(2), size(h{1},3))),...
             single(zeros(use_sz(1), use_sz(2), size(h{2},3))),...
             single(zeros(use_sz(1), use_sz(2), size(h{3},3))),...
             single(zeros(use_sz(1), use_sz(2), size(h{4},3)))};
        t{1}(sx{1},sy{1},:) = h{1};
        t{2}(sx{2},sy{2},:) = h{2};
        t{3}(sx{3},sy{3},:) = h{3};
        t{4}(sx{4},sy{4},:) = h{4};

        h_f = {fft2(t{1}), fft2(t{2}), fft2(t{3}), fft2(t{4})};
        %   update L
        l_f = {l_f{1} + (mu{1} * (g_f{1} - h_f{1})), l_f{2} + (mu{2} * (g_f{2} - h_f{2})), l_f{3} + (mu{3} * (g_f{3} - h_f{3})), l_f{4} + (mu{4} * (g_f{4} - h_f{4}))};
        %   update mu- betha = 10.
        mu = {min(betha{1} * mu{1}, mumax{1}), min(betha{2} * mu{2}, mumax{2}), min(betha{3} * mu{3}, mumax{3}), min(betha{4} * mu{4}, mumax{4})};
        i = i+1;
    end
    
    target_sz = floor(base_target_sz * currentScaleFactor);
    
    %save position and calculate FPS
    rect_position(loop_frame,:) = [pos([2,1]) - floor(target_sz([2,1])/2), target_sz([2,1])];
    
    time = time + toc();
    
    %visualization
    if visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        ground_truth_original=ground(frame,:);
%         ground_truth_original_bacf = ground_truth_bacf(frame,:);
%         ground_truth_original_kcf = ground_truth_kcf(frame,:);
%         ground_truth_original_ldes = ground_truth_ldes(frame,:);
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if frame == 1
            fig_handle = figure('Name', 'Tracking');
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else
            resp_sz = round(sz*currentScaleFactor*scaleFactors(scale_ind));
            xs = floor(old_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
            ys = floor(old_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
            sc_ind = floor((nScales - 1)/2) + 1;
            
            figure(fig_handle);
            imagesc(im_to_show);
            hold on;

            %resp_handle = imagesc(xs, ys, fftshift(response(:,:,sc_ind))); colormap hsv;
            %alpha(resp_handle, 0.2);
            
%             rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
%             text(20, 30, ['Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [1 0 0], 'fontsize', 16);
%             text(20, 60, ['FPS : ' num2str(1/(time/loop_frame))], 'color', [1 0 0], 'fontsize', 16);
           rectangle('Position',rect_position_vis, 'EdgeColor','y', 'LineWidth',5);
            rectangle('Position',ground_truth_original, 'EdgeColor','r', 'LineWidth',5);
%             rectangle('Position',ground_truth_original_bacf, 'EdgeColor','g', 'LineWidth',5);
%             rectangle('Position',ground_truth_original_kcf, 'EdgeColor','b', 'LineWidth',5);
%             rectangle('Position',ground_truth_original_ldes, 'EdgeColor','k', 'LineWidth',5);
            %text(20, 30, ['# Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 20);
            text(20, 30, ['#' int2str(loop_frame)], 'color', [1 0 0], 'fontsize', 25);
            hold off;
        end
        drawnow
        
    end
%     picname=[num2str(frame) '.jpg'];
%     hold on;
%     saveas(gcf,picname);    
%     pause(0.5);
    loop_frame = loop_frame + 1;
end
close(figure(fig_handle));
%   save resutls
fps = loop_frame / time;
results.type = 'rect';
results.res = rect_position;
results.fps = fps;
