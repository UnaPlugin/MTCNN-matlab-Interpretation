# MTCNN-matlab-Interpretation
Interpretation of the MTCNN (2016) codes in matlab.


# MTCNN的Matlab源码读、注释

## detect_face.m

    function [total_boxes points] = detect_face(img,minsize,PNet,RNet,ONet,threshold,fastresize,factor)
    	%im: input image
    	%minsize: minimum of faces' size	%%取40
    	%pnet, rnet, onet: caffemodel
    	%threshold: threshold=[th1 th2 th3], th1-3 are three steps's threshold
    	%fastresize: resize img from last scale (using in high-resolution images) if fastresize==true

    	factor_count=0;
    	total_boxes=[];
    	points=[];
    	h=size(img,1);
    	w=size(img,2);
    	minl=min([w h]);			%%短边，假设480x640则为480
    	img=single(img);
    	if fastresize
    		im_data=(single(img)-127.5)*0.0078125;	//映射至-1~1
    	end
    	m=12/minsize;				%%12÷40=0.3
    	minl=minl*m;				%%480x0.3=144
		***********************************************************************************************
    	%creat scale pyramid 降采样
    	scales=[];
    	while (minl>=12)			%%其实这个12没有什么意义，可能是为了两边计算的浮点数的小数位数少一些【1】
    		scales=[scales m*factor^(factor_count)];//【1】式中的x即为factor_count，scales存储已经采样过的scale
    		minl=minl*factor;		%%每轮x0.7,降采样
    		factor_count=factor_count+1;
    	end
		***********************************************************************************************
    	%first stage 第一层PNet
    	for j = 1:size(scales,2)%%因为是按行存储的，所以获取size要按列数获取
    		scale=scales(j);
    		hs=ceil(h*scale);	%%h×scale的向上取整,resize过程
    		ws=ceil(w*scale);	%%w×scale的向上取整,resize过程
    		if fastresize
    			im_data=imResample(im_data,[hs ws],'bilinear');	%%降采样，相当于resize
    		else 
    			im_data=(imResample(img,[hs ws],'bilinear')-127.5)*0.0078125;%%降采样+归一化
 		end
    		PNet.blobs('data').reshape([hs ws 3 1]);			%%PNet是在test.m中直接导入的Caffe的Net，这里重新reshap了data这个blob（节点）的维数
    		out=PNet.forward({im_data});
    		boxes=generateBoundingBox(out{2}(:,:,2),out{1},scale,threshold(1));%%作者单独写的function，就是产生boundingbox的函数，见下文%%threshold(1)是PNet的阈值
    		%inter-scale nms
    		pick=nms(boxes,0.5,'Union');			%%非最大值抑制，作者单独写的function
    		boxes=boxes(pick,:);					%%？？？将符合要求的box保留下来，存成列向量给到boxes里
    		if ~isempty(boxes)						%%防止加上空box
    			total_boxes=[total_boxes;boxes];	%%按列加上去新的box
    		end
    	end
    	numbox=size(total_boxes,1);	%%因为box是按列存储的，所以box数为行数，[x1,y1,x2,y2],(这里的x、y均为列向量)
    	if ~isempty(total_boxes)	%%total_boxes是矩阵（？）
    		pick=nms(total_boxes,0.7,'Union');		%%做非第二次最大值抑制（？），作者单独写的function
    		total_boxes=total_boxes(pick,:);		%%？？？将符合要求的box保留下来，
    		regw=total_boxes(:,3)-total_boxes(:,1);	%%第三列的减去第一列的元素，得到[w](列向量)
    		regh=total_boxes(:,4)-total_boxes(:,2);	%%第四列的减去第二列的元素，得到[h](列向量)
    		total_boxes=[total_boxes(:,1)+total_boxes(:,6).*regw total_boxes(:,2)\
									     +total_boxes(:,7).*regh total_boxes(:,3)\
										 +total_boxes(:,8).*regw total_boxes(:,4)\
										 +total_boxes(:,9).*regh total_boxes(:,5)];	%%TODO:这段totalboxes没懂
    		total_boxes=rerec(total_boxes);			%%rerec是作者写的function，用于把boxes变成方形（？），见下文
    		total_boxes(:,1:4)=fix(total_boxes(:,1:4));	%%fix为向零取整
    		[dy edy dx edx y ey x ex tmpw tmph]=pad(total_boxes,w,h);	%%作者自己写的function
    	end
    	numbox=size(total_boxes,1);
    
    	***********************************************************************************************
    	if numbox>0
    		%second stage 第二层RNet
     		tempimg=zeros(24,24,3,numbox);
    		for k=1:numbox
    			tmp=zeros(tmph(k),tmpw(k),3);	%%维数为（height,width,3（通道？））
    			tmp(dy(k):edy(k),dx(k):edx(k),:)=img(y(k):ey(k),x(k):ex(k),:);//dy,y,edy,ey等都是PNet中pad得到的boxes后得到的
    			tempimg(:,:,:,k)=imResample(tmp,[24 24],'bilinear');	//降采样至24x24的size
    		end
    		tempimg=(tempimg-127.5)*0.0078125;	%%归一化
    		RNet.blobs('data').reshape([24 24 3 numbox]);	%%（height,width,channels,num）
    		out=RNet.forward({tempimg});	%%前向计算，得到output，RNet这个caffenet在test.m中已经导入好了%%TODO:out的格式还需要查一下
    		score=squeeze(out{2}(2,:));		%%squeeze是将1x1xN的多维向量变成N个元素的单维向量
    		pass=find(score>threshold(2));	%%threshold(2)是RNet的阈值
    		total_boxes=[total_boxes(pass,1:4) score(pass)'];	%%score(pass)'将score变成列向量
    		mv=out{1}(:,pass);				%%TODO:out的格式暂时未知
    		if size(total_boxes,1)>0		
    			pick=nms(total_boxes,0.7,'Union');
    			total_boxes=total_boxes(pick,:); 
    			total_boxes=bbreg(total_boxes,mv(:,pick)');	%bbreg是作者写的function，校准框框，见下文
    			total_boxes=rerec(total_boxes);	%%rerec是作者写的function，用于把boxes变成方形（？），见下文
    		end
    		numbox=size(total_boxes,1);
    
    	***********************************************************************************************
    		if numbox>0
    			%third stage 第三层ONet
    			total_boxes=fix(total_boxes);
    			[dy edy dx edx y ey x ex tmpw tmph]=pad(total_boxes,w,h);	%%先pad一下
    			tempimg=zeros(48,48,3,numbox);
    			for k=1:numbox
    				tmp=zeros(tmph(k),tmpw(k),3);
    				tmp(dy(k):edy(k),dx(k):edx(k),:)=img(y(k):ey(k),x(k):ex(k),:);%%原图的dataclone至tmp
    				tempimg(:,:,:,k)=imResample(tmp,[48 48],'bilinear');	%%统一降采样至48x48的size
    			end
    			tempimg=(tempimg-127.5)*0.0078125;	%%归一化
    			ONet.blobs('data').reshape([48 48 3 numbox]);	%%把data这个blob的size调整至48x48x3xnum个
    			out=ONet.forward({tempimg});
    			score=squeeze(out{3}(2,:));
    			points=out{2};						%%TODO:
    			pass=find(score>threshold(3));
    			points=points(:,pass);				%%选取大于阈值的
    			total_boxes=[total_boxes(pass,1:4) score(pass)'];
    			mv=out{1}(:,pass);
    			w=total_boxes(:,3)-total_boxes(:,1)+1;	%%之前的地方没有+1处理（我也不知道为啥）
    			h=total_boxes(:,4)-total_boxes(:,2)+1;
    			points(1:5,:)=repmat(w',[5 1]).*points(1:5,:)+repmat(total_boxes(:,1)',[5 1])-1;	%%repmat是堆叠矩阵函数，B=repmat(A,[a b]),堆叠A这个矩阵，在行数上，堆叠a次，在列数上堆叠b次，得到B
    			points(6:10,:)=repmat(h',[5 1]).*points(6:10,:)+repmat(total_boxes(:,2)',[5 1])-1;


    			if size(total_boxes,1)>0				
    				total_boxes=bbreg(total_boxes,mv(:,:)');	%%校准框框，也是做的一个筛选
    				pick=nms(total_boxes,0.7,'Min');	%%0.7为阈值
    				total_boxes=total_boxes(pick,:);  				
    				points=points(:,pick);
    			end
    		end
    	end 	
    end
    
【1】       ![](https://latex.codecogs.com/gif.download?%5Cfrac%7B12%7D%7Bminsize%7D%5Ccdot%20minl%5Ccdot%20%280.7%29%5E%7Bx%7D%5Cgeqslant%2012)


## nms.m 非最大值抑制的function
    function pick = nms(boxes,threshold,type)
    	%NMS
    	if isempty(boxes)
    	  pick = [];
    	  return;
    	end
    	x1 = boxes(:,1);
    	y1 = boxes(:,2);
    	x2 = boxes(:,3);
    	y2 = boxes(:,4);
    	s = boxes(:,5);
    	area = (x2-x1+1) .* (y2-y1+1);
    	[vals, I] = sort(s);	%%对s做个升序排列，I为索引数组（就是A中元素在B中怎么排列的序号的矩阵），即有：B(:,j) = A(I(:,j),j)
    	pick = s*0;		%%保证pick的尺寸和s一样
    	counter = 1;
    	while ~isempty(I)
    		last = length(I);
    		i = I(last);	%%得到I的最后一个元素值，即最大元素的那个索引值
    		pick(counter) = i;	%%这个索引值存给pick
    		counter = counter + 1;  
    		xx1 = max(x1(i), x1(I(1:last-1)));	%%左上角这个点，选最大的，即尽量往右下角靠拢
    		yy1 = max(y1(i), y1(I(1:last-1)));	%%左上角这个点，选最大的，即尽量往右下角靠拢
    		xx2 = min(x2(i), x2(I(1:last-1)));	%%右下角这个点，选最小的，即尽量往左上角靠拢
    		yy2 = min(y2(i), y2(I(1:last-1)));  %%右下角这个点，选最小的，即尽量往左上角靠拢
    		w = max(0.0, xx2-xx1+1);	%%保证不要减到负数，引起报错
    		h = max(0.0, yy2-yy1+1); 	%%保证不要减到负数，引起报错
    		inter = w.*h;
    		if strcmp(type,'Min')
    			o = inter ./ min(area(i),area(I(1:last-1)));	%%inter算出来的面积除以最小的框的面积
    		else
    			o = inter ./ (area(i) + area(I(1:last-1)) - inter);	%%另一种算法，分母可能会比上一种的分母大一些
    		end
    		I = I(find(o<=threshold));	%%小于0.7的元素的索引值保留
    	end
    	pick = pick(1:(counter-1));	%%重新再赋一遍值（有什么意义吗？）
    end

## pad.m （等等还没看懂）填充
####比较重要的一个函数

    function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
    	%compute the padding coordinates (pad the bounding boxes to square)
    	tmpw=total_boxes(:,3)-total_boxes(:,1)+1;	%%[w]所有box的宽度列向量 %%有时候作者会不做这个+1处理
    	tmph=total_boxes(:,4)-total_boxes(:,2)+1;	%%[h]所有box的宽度列向量
    	numbox=size(total_boxes,1);
    	
    	dx=ones(numbox,1);dy=ones(numbox,1);		%%dx,dy均为列向量
    	edx=tmpw;edy=tmph;	
    	
    	x=total_boxes(:,1);y=total_boxes(:,2);		%%x是左上角点的x，y是左上角点的y
    	ex=total_boxes(:,3);ey=total_boxes(:,4);	%%ex是右下角点的x，ey是右下角点的y
    	
    	tmp=find(ex>w);	%%保证逻辑上对（右下角点的坐标一定要比宽（长）大）
    	edx(tmp)=-ex(tmp)+w+tmpw(tmp);ex(tmp)=w;	%%	
    	
    	tmp=find(ey>h);
    	edy(tmp)=-ey(tmp)+h+tmph(tmp);ey(tmp)=h;	
    	
    	tmp=find(x<1);
    	dx(tmp)=2-x(tmp);x(tmp)=1;	
    	
    	tmp=find(y<1);
    	dy(tmp)=2-y(tmp);y(tmp)=1;
    end



## generateBoundingBox.m	框个box

    function [boundingbox reg] = generateBoundingBox(map,reg,scale,t)
    	%use heatmap to generate bounding boxes
    	stride=2;
    	cellsize=12;
    	boundingbox=[];
    	map=map';
    	dx1=reg(:,:,1)';
    	dy1=reg(:,:,2)';
    	dx2=reg(:,:,3)';
    	dy2=reg(:,:,4)';
    	[y x]=find(map>=t);
    	a=find(map>=t); 
    	if size(y,1)==1
    		y=y';x=x';score=map(a)';dx1=dx1';dy1=dy1';dx2=dx2';dy2=dy2';
    	else
    		score=map(a);
    	end   
    	reg=[dx1(a) dy1(a) dx2(a) dy2(a)];
    	if isempty(reg)
    		reg=reshape([],[0 3]);
    	end
    	boundingbox=[y x];
    	boundingbox=[fix((stride*(boundingbox-1)+1)/scale) fix((stride*(boundingbox-1)+cellsize-1+1)/scale) score reg];
    end

## bbreg.m    校准框框
    function [boundingbox] = bbreg(boundingbox,reg)
    	%calibrate bouding boxes
    	if size(reg,2)==1
    		reg=reshape(reg,[size(reg,3) size(reg,4)])';	%没懂这个resize啥意思
    	end
    	w=[boundingbox(:,3)-boundingbox(:,1)]+1;
    	h=[boundingbox(:,4)-boundingbox(:,2)]+1;
    	boundingbox(:,1:4)=[boundingbox(:,1)+reg(:,1).*w boundingbox(:,2)+reg(:,2).*h boundingbox(:,3)+reg(:,3).*w boundingbox(:,4)+reg(:,4).*h];
    end

## rerec.m
    
    function [bboxA] = rerec(bboxA)
    	%convert bboxA to square
    	bboxB=bboxA(:,1:4);
    	h=bboxA(:,4)-bboxA(:,2);
    	w=bboxA(:,3)-bboxA(:,1);
    	l=max([w h]')';
    	bboxA(:,1)=bboxA(:,1)+w.*0.5-l.*0.5;
    	bboxA(:,2)=bboxA(:,2)+h.*0.5-l.*0.5;
    	bboxA(:,3:4)=bboxA(:,1:2)+repmat(l,[1 2]);
    end


## test.m主程序

    function varargout = test(varargin)
    gui_Singleton = 1;
    gui_State = struct('gui_Name',   mfilename, ...
       'gui_Singleton',  gui_Singleton, ...
       'gui_OpeningFcn', @test_OpeningFcn, ...
       'gui_OutputFcn',  @test_OutputFcn, ...
       'gui_LayoutFcn',  [] , ...
       'gui_Callback',   []);
    if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
    end
    if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
    else
    gui_mainfcn(gui_State, varargin{:});
    end
    
    
    
    
    function test_OpeningFcn(hObject, eventdata, handles, varargin)
    handles.output = hObject;
    guidata(hObject, handles);
    function varargout = test_OutputFcn(hObject, eventdata, handles) 
    varargout{1} = handles.output;
    function pushbutton1_Callback(hObject, eventdata, handles)
    threshold=[0.6 0.7 str2num(get(findobj('tag','edit4'),'string'))]
	factor=0.709;
    minsize=str2num(get(findobj('tag','edit6'),'string'));
    stop=0.03;
    mypath=get(findobj('tag','edit1'),'string')
    addpath(mypath);
    caffe.reset_all();
    caffe.set_mode_cpu();
    mypath=get(findobj('tag','edit3'),'string')
    addpath(mypath);
    cameraid=get(findobj('tag','edit2'),'string')
    camera=imaqhwinfo;
    camera=camera.InstalledAdaptors{str2num(cameraid)}
    vid1= videoinput(camera,1,get(findobj('tag','edit5'),'string'));
    warning off all    
    usbVidRes1=get(vid1,'videoResolution');
    nBands1=get(vid1,'NumberOfBands');
    hImage1=imshow(zeros(usbVidRes1(2),usbVidRes1(1),nBands1));
    preview(vid1,hImage1);
    
    prototxt_dir = './model/det1.prototxt';
    model_dir = './model/det1.caffemodel';
    PNet=caffe.Net(prototxt_dir,model_dir,'test');	//直接导入的caffe网络
	prototxt_dir = './model/det2.prototxt';
    model_dir = './model/det2.caffemodel';
    RNet=caffe.Net(prototxt_dir,model_dir,'test');
	prototxt_dir = './model/det3.prototxt';
    model_dir = './model/det3.caffemodel';
    ONet=caffe.Net(prototxt_dir,model_dir,'test');  
    prototxt_dir = './model/det4.prototxt';
    model_dir = './model/det4.caffemodel';
    LNet=caffe.Net(prototxt_dir,model_dir,'test');  
    rec=rectangle('Position',[1 1 1 1],'Edgecolor','r');
    while (1)
        img=getsnapshot(vid1);
        [total_boxes point]=detect_face(img,minsize,PNet,RNet,ONet,threshold,false,factor);
        try
            delete(rec);
        catch
        end
        numbox=size(total_boxes,1);
        for j=1:numbox;       
            rec(j)=rectangle('Position',[total_boxes(j,1:2) total_boxes(j,3:4)-total_boxes(j,1:2)],'Edgecolor','g','LineWidth',3);
            rec(6*numbox+j)=rectangle('Position',[point(1,j),point(6,j),5,5],'Curvature',[1,1],'FaceColor','g','LineWidth',3);
            rec(12*numbox+j)=rectangle('Position',[point(2,j),point(7,j),5,5],'Curvature',[1,1],'FaceColor','g','LineWidth',3);
            rec(18*numbox+j)=rectangle('Position',[point(3,j),point(8,j),5,5],'Curvature',[1,1],'FaceColor','g','LineWidth',3);
            rec(24*numbox+j)=rectangle('Position',[point(4,j),point(9,j),5,5],'Curvature',[1,1],'FaceColor','g','LineWidth',3);
            rec(30*numbox+j)=rectangle('Position',[point(5,j),point(10,j),5,5],'Curvature',[1,1],'FaceColor','g','LineWidth',3);
        end
        pause(stop)
    end



    function edit1_Callback(hObject, eventdata, handles)
    % hObjecthandle to edit1 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handlesstructure with handles and user data (see GUIDATA)
    
    % Hints: get(hObject,'String') returns contents of edit1 as text
    %str2double(get(hObject,'String')) returns contents of edit1 as a double
    
    
    % --- Executes during object creation, after setting all properties.
    function edit1_CreateFcn(hObject, eventdata, handles)
    % hObjecthandle to edit1 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handlesempty - handles not created until after all CreateFcns called
    
    % Hint: edit controls usually have a white background on Windows.
    %   See ISPC and COMPUTER.
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
    end
    
    
    
    function edit2_Callback(hObject, eventdata, handles)
    % hObjecthandle to edit2 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handlesstructure with handles and user data (see GUIDATA)
    
    % Hints: get(hObject,'String') returns contents of edit2 as text
    %str2double(get(hObject,'String')) returns contents of edit2 as a double
    
    
    % --- Executes during object creation, after setting all properties.
    function edit2_CreateFcn(hObject, eventdata, handles)
    % hObjecthandle to edit2 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handlesempty - handles not created until after all CreateFcns called
    
    % Hint: edit controls usually have a white background on Windows.
    %   See ISPC and COMPUTER.
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
    end
    
    
    
    function edit3_Callback(hObject, eventdata, handles)
    % hObjecthandle to edit3 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handlesstructure with handles and user data (see GUIDATA)
    
    % Hints: get(hObject,'String') returns contents of edit3 as text
    %str2double(get(hObject,'String')) returns contents of edit3 as a double
    
    
    % --- Executes during object creation, after setting all properties.
    function edit3_CreateFcn(hObject, eventdata, handles)
    % hObjecthandle to edit3 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handlesempty - handles not created until after all CreateFcns called
    
    % Hint: edit controls usually have a white background on Windows.
    %   See ISPC and COMPUTER.
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
    end
    
    
    
    function edit4_Callback(hObject, eventdata, handles)
    % hObjecthandle to edit4 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handlesstructure with handles and user data (see GUIDATA)
    
    % Hints: get(hObject,'String') returns contents of edit4 as text
    %str2double(get(hObject,'String')) returns contents of edit4 as a double
    
    
    % --- Executes during object creation, after setting all properties.
    function edit4_CreateFcn(hObject, eventdata, handles)
    % hObjecthandle to edit4 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handlesempty - handles not created until after all CreateFcns called
    
    % Hint: edit controls usually have a white background on Windows.
    %   See ISPC and COMPUTER.
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
    end
    
    
    
    function edit5_Callback(hObject, eventdata, handles)
    % hObjecthandle to edit5 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handlesstructure with handles and user data (see GUIDATA)
    
    % Hints: get(hObject,'String') returns contents of edit5 as text
    %str2double(get(hObject,'String')) returns contents of edit5 as a double
    
    
    % --- Executes during object creation, after setting all properties.
    function edit5_CreateFcn(hObject, eventdata, handles)
    % hObjecthandle to edit5 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handlesempty - handles not created until after all CreateFcns called
    
    % Hint: edit controls usually have a white background on Windows.
    %   See ISPC and COMPUTER.
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
    end
    
    
    
    function edit6_Callback(hObject, eventdata, handles)
    % hObjecthandle to edit6 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handlesstructure with handles and user data (see GUIDATA)
    
    % Hints: get(hObject,'String') returns contents of edit6 as text
    %str2double(get(hObject,'String')) returns contents of edit6 as a double
    
    
    % --- Executes during object creation, after setting all properties.
    function edit6_CreateFcn(hObject, eventdata, handles)
    % hObjecthandle to edit6 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handlesempty - handles not created until after all CreateFcns called
    
    % Hint: edit controls usually have a white background on Windows.
    %   See ISPC and COMPUTER.
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
    end
