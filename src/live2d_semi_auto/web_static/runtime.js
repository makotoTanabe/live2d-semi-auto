/* Editable, dependency-free Canvas2D character mesh runtime. No Cubism SDK. */
(function (global) {
  'use strict';
  const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));
  const ranges = {angleX:[-1,1], angleY:[-1,1], angleZ:[-1,1], eyeOpen:[0,1], mouthOpen:[0,1], breath:[0,1]};
  const expressionNames = ['neutral','smile','angry','cry','surprised','shy'];
  const motionNames = ['idle','nod','shake','greeting'];
  const headRoles = new Set(['face','hair_back','hair_front','hair_side','eye','iris','eyebrow','mouth','accessory_head']);
  function number(value, fallback) { return typeof value === 'number' && Number.isFinite(value) ? value : fallback; }
  function point(value, fallback, w, h) {
    return Array.isArray(value) && value.length === 2 && value.every(v => typeof v === 'number' && Number.isFinite(v))
      ? [clamp(value[0],0,w),clamp(value[1],0,h)] : fallback.slice();
  }
  function triangle(ctx, image, src, dst) {
    const [s0,s1,s2] = src, [d0,d1,d2] = dst;
    const x1=s1[0]-s0[0], y1=s1[1]-s0[1], x2=s2[0]-s0[0], y2=s2[1]-s0[1];
    const det=x1*y2-x2*y1;
    if (Math.abs(det)<1e-7) return;
    const u1=d1[0]-d0[0], v1=d1[1]-d0[1], u2=d2[0]-d0[0], v2=d2[1]-d0[1];
    const a=(u1*y2-u2*y1)/det, c=(u2*x1-u1*x2)/det;
    const b=(v1*y2-v2*y1)/det, d=(v2*x1-v1*x2)/det;
    ctx.save(); ctx.beginPath();
    for (let i=0;i<3;i++) {
      if(i===0)ctx.moveTo(...dst[i]);else ctx.lineTo(...dst[i]);
    }
    ctx.closePath(); ctx.clip();
    ctx.transform(a,b,c,d,d0[0]-a*s0[0]-c*s0[1],d0[1]-b*s0[0]-d*s0[1]);
    ctx.drawImage(image,0,0); ctx.restore();
  }
  class Character {
    constructor(canvas, model, options={}) {
      if (!canvas || typeof canvas.getContext!=='function') throw new Error('描画先のcanvasが必要です。');
      if (!model || model.version!==1 || !Array.isArray(model.canvas) || model.canvas.length!==2 ||
          !model.canvas.every(v=>Number.isFinite(v)&&v>0&&v<=30000) || !Array.isArray(model.layers))
        throw new Error('Webモデルの形式が不正です。');
      this.canvas=canvas; this.ctx=canvas.getContext('2d');
      if (!this.ctx) throw new Error('Canvas2Dを利用できません。');
      this.model=model; this.width=model.canvas[0]; this.height=model.canvas[1]; this.options=options;
      this.layers=[]; this.loaded=false; this.destroyed=false; this.running=false; this._frame=null;
      this._meshCanvas=document.createElement('canvas');this._meshContext=this._meshCanvas.getContext('2d');
      this.parameters={angleX:0,angleY:0,angleZ:0,eyeOpen:1,mouthOpen:0,breath:0};
      this._effective={...this.parameters}; this.expression='neutral'; this.motion='idle';
      this.autoBlink=options.autoBlink!==false; this.idle=options.idle!==false;
      this.pointerTracking=options.pointerTracking!==false; this.pointer={x:0,y:0,active:false};
      this.strength=clamp(number(model.config && model.config.strength,1),0,2);
      this.resolution=Math.round(clamp(number(model.config && model.config.meshResolution,4),2,8));
      this.headPivot=point(model.groups && model.groups.head && model.groups.head.pivot,[this.width*.5,this.height*.38],this.width,this.height);
      this.bodyPivot=point(model.groups && model.groups.body && model.groups.body.pivot,[this.width*.5,this.height*.82],this.width,this.height);
      this._move=event=>{
        if(!this.pointerTracking)return;
        const bounds=this.canvas.getBoundingClientRect();
        if(bounds.width<=0||bounds.height<=0)return;
        this.pointer={x:clamp((event.clientX-bounds.left)/bounds.width*2-1,-1,1),
                      y:clamp((event.clientY-bounds.top)/bounds.height*2-1,-1,1),active:true};
      };
      this._leave=()=>{this.pointer.active=false;};
      canvas.addEventListener('pointermove',this._move); canvas.addEventListener('pointerleave',this._leave);
    }
    async load() {
      if(this.destroyed)throw new Error('破棄済みのモデルは読み込めません。');
      const ids=new Set();
      const layers=this.model.layers.slice().sort((a,b)=>number(a.z_order,0)-number(b.z_order,0));
      const loaded=await Promise.all(layers.map(async layer=>{
        if(!layer || typeof layer.id!=='string' || ids.has(layer.id) || !Array.isArray(layer.bounds) || layer.bounds.length!==4 ||
           !layer.bounds.every(Number.isFinite) || layer.bounds[0]<0 || layer.bounds[1]<0 ||
           layer.bounds[2]<=layer.bounds[0] || layer.bounds[3]<=layer.bounds[1] ||
           layer.bounds[2]>this.width || layer.bounds[3]>this.height)throw new Error('パーツ情報が不正です。');
        ids.add(layer.id);
        const assetUrls=this.options.assetUrls || {};
        let url=Object.prototype.hasOwnProperty.call(assetUrls,layer.id) ? assetUrls[layer.id] : null;
        if(url===null) {
          // Bundle paths stay relative; remote assets must be explicitly supplied.
          if(typeof layer.asset!=='string' || !/^parts\/[a-zA-Z0-9_-]+\.png$/.test(layer.asset))
            throw new Error('パーツ画像のパスが不正です。');
          url=(this.options.baseUrl || '')+layer.asset;
        }
        if(typeof url!=='string' || /^\s*(javascript|vbscript):/i.test(url))throw new Error('パーツ画像のURLが不正です。');
        const image=new Image();
        if(this.options.crossOrigin)image.crossOrigin=this.options.crossOrigin;
        await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(new Error('パーツ画像を読み込めません。'));image.src=url;});
        if(image.naturalWidth!==layer.bounds[2]-layer.bounds[0] || image.naturalHeight!==layer.bounds[3]-layer.bounds[1])
          throw new Error('パーツ画像と配置寸法が一致しません。');
        return {...layer,image,pivot:point(layer.pivot,[(layer.bounds[0]+layer.bounds[2])/2,layer.bounds[1]],this.width,this.height)};
      }));
      if(this.destroyed)return this;
      this.layers=loaded;this.loaded=true;this.render(0);return this;
    }
    start() {
      if(this.running||this.destroyed)return this;
      this.running=true;
      const step=timestamp=>{if(!this.running||this.destroyed)return;this.render(timestamp);this._frame=requestAnimationFrame(step);};
      this._frame=requestAnimationFrame(step);return this;
    }
    stop() {this.running=false;if(this._frame!==null)cancelAnimationFrame(this._frame);this._frame=null;return this;}
    destroy() {this.stop();this.canvas.removeEventListener('pointermove',this._move);this.canvas.removeEventListener('pointerleave',this._leave);this.layers=[];this._meshCanvas.width=this._meshCanvas.height=1;this.loaded=false;this.destroyed=true;}
    resize(width,height) {
      if(!Number.isFinite(width)||!Number.isFinite(height)||width<1||height<1)throw new Error('キャンバス寸法が不正です。');
      this.canvas.width=Math.round(clamp(width,1,8192));this.canvas.height=Math.round(clamp(height,1,8192));return this;
    }
    setParameters(values) {
      if(!values||typeof values!=='object')throw new Error('パラメータはオブジェクトで指定してください。');
      for(const [key,value] of Object.entries(values)) {
        if(!Object.prototype.hasOwnProperty.call(ranges,key))throw new Error('未対応のパラメータです。');
        if(typeof value!=='number'||!Number.isFinite(value))throw new Error('パラメータは有限の数値で指定してください。');
      }
      for(const [key,value] of Object.entries(values))this.parameters[key]=clamp(value,...ranges[key]);
      return this;
    }
    getParameters() {return {...this.parameters};}
    getEffectiveParameters() {return {...this._effective};}
    setExpression(name) {if(!expressionNames.includes(name))throw new Error('表情名が不正です。');this.expression=name;return this;}
    setMotion(name) {if(!motionNames.includes(name))throw new Error('モーション名が不正です。');this.motion=name;return this;}
    playMotion(name) {return this.setMotion(name);}
    setAutoBlink(value) {this.autoBlink=Boolean(value);return this;}
    setIdle(value) {this.idle=Boolean(value);return this;}
    setPointerTracking(value) {this.pointerTracking=Boolean(value);if(!this.pointerTracking)this.pointer.active=false;return this;}
    _expression() {
      const defaultPresets={neutral:{eye:1,mouth:0,smile:0,brow:0,blush:0,tear:0},smile:{eye:.7,mouth:.3,smile:1,brow:.2,blush:.15,tear:0},angry:{eye:.7,mouth:.25,smile:-.8,brow:-1,blush:.15,tear:0},cry:{eye:.5,mouth:.65,smile:-1,brow:.8,blush:.1,tear:1},surprised:{eye:1.15,mouth:.9,smile:0,brow:1,blush:0,tear:0},shy:{eye:.75,mouth:.15,smile:.5,brow:.4,blush:1,tear:0}};
      const input=(this.model.expressions || {})[this.expression] || defaultPresets[this.expression];
      const fallback=defaultPresets[this.expression], result={};
      for(const key of Object.keys(fallback))result[key]=clamp(number(input[key],fallback[key]),key==='smile'||key==='brow'?-1:0,key==='eye'?1.2:1);
      return result;
    }
    _frameParameters(seconds) {
      const p={...this.parameters};
      if(this.idle) {p.angleX+=Math.sin(seconds*.7)*.07;p.angleZ+=Math.sin(seconds*.9)*.06;p.breath=Math.max(p.breath,(Math.sin(seconds*1.8)+1)*.5);}
      if(this.pointerTracking&&this.pointer.active){p.angleX+=this.pointer.x*.8;p.angleY+=this.pointer.y*.65;p.angleZ+=this.pointer.x*.15;}
      if(this.motion==='nod')p.angleY+=Math.sin(seconds*3)*.7;
      if(this.motion==='shake')p.angleX+=Math.sin(seconds*3.3)*.8;
      if(this.motion==='greeting'){p.angleZ+=Math.sin(seconds*2)*.12;p.angleY+=Math.sin(seconds*2)*.2;}
      if(this.autoBlink) {
        const phase=((seconds%4.7)+4.7)%4.7;
        if(phase>3.9&&phase<4.12)p.eyeOpen*=Math.abs((phase-4.01)/.11);
      }
      for(const key of Object.keys(p))p[key]=clamp(p[key],...ranges[key]);
      this._effective={...p};return p;
    }
    _bodyPoint(x,y,p) {
      // Breathing is a shared affine group deformation around the body pivot.
      // Every layer and the neck anchor follows the same expansion, so clothing
      // remains seamless and does not acquire artificial mesh-grid joints.
      return [this.bodyPivot[0]+(x-this.bodyPivot[0])*(1+p.breath*.006*this.strength),
              this.bodyPivot[1]+(y-this.bodyPivot[1])*(1+p.breath*.009*this.strength)];
    }
    _headPoint(x,y,p) {
      const pivot=this._bodyPoint(...this.headPivot,p);
      const dx=x-this.headPivot[0], dy=y-this.headPivot[1];
      const angle=p.angleZ*.12*this.strength, cos=Math.cos(angle),sin=Math.sin(angle);
      const xx=dx*(1-Math.abs(p.angleX)*.06*this.strength), yy=dy*(1-Math.abs(p.angleY)*.035*this.strength);
      return [pivot[0]+xx*cos-yy*sin+p.angleX*this.width*.03*this.strength,
              pivot[1]+xx*sin+yy*cos+p.angleY*this.height*.013*this.strength];
    }
    _vertex(layer,x,y,p,expression,seconds) {
      const bounds=layer.bounds, role=layer.role;
      const cx=(bounds[0]+bounds[2])/2,cy=(bounds[1]+bounds[3])/2;
      const w=bounds[2]-bounds[0],h=bounds[3]-bounds[1];
      if(role==='eye'||role==='iris') {
        const open=Math.max(.025,p.eyeOpen*expression.eye);
        y=cy+(y-cy)*open;
        if(role==='iris')x+=p.angleX*w*.08*this.strength;
      }
      if(role==='mouth') {
        const open=Math.max(p.mouthOpen,expression.mouth);
        x=cx+(x-cx)*(1+expression.smile*.12);
        y=cy+(y-cy)*(1+open*1.5)-expression.smile*h*.32*Math.pow(Math.abs((x-cx)/Math.max(1,w*.5)),2);
      }
      if(role==='eyebrow') {
        const side=cx<this.headPivot[0]?-1:1;
        y-=expression.brow*h*.9;y+=(x-cx)*expression.brow*.12*side;
      }
      if(role.startsWith('hair_')) {
        const weight=Math.pow(clamp((y-bounds[1])/Math.max(1,h),0,1),2);
        const phase=role==='hair_back'?.8:role==='hair_side'?1.5:0;
        const sway=(this.idle?Math.sin(seconds*2.1+phase)*.004:0)+p.angleX*.006;
        x+=sway*this.width*weight*this.strength;
      }
      if(role==='arm_left'||role==='arm_right') {
        const wave=this.motion==='greeting'&&role==='arm_right'?(.65+Math.sin(seconds*4)*.3):0;
        const pivot=layer.pivot,angle=wave*.65*this.strength;
        const dx=x-pivot[0],dy=y-pivot[1];
        x=pivot[0]+dx*Math.cos(angle)-dy*Math.sin(angle);
        y=pivot[1]+dx*Math.sin(angle)+dy*Math.cos(angle);
      }
      if(headRoles.has(role))return this._headPoint(x,y,p);
      if(role==='static')return [x,y];
      return this._bodyPoint(x,y,p);
    }
    _mouthInterior(layer,p,expression) {
      const open=Math.max(p.mouthOpen,expression.mouth);
      if(open<.15)return;
      const bounds=layer.bounds,w=bounds[2]-bounds[0],h=bounds[3]-bounds[1];
      const center=this._headPoint((bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2,p);
      const ctx=this.ctx;ctx.save();ctx.translate(...center);ctx.rotate(p.angleZ*.12*this.strength);
      ctx.fillStyle='#672631';ctx.beginPath();ctx.ellipse(0,0,w*.32,h*(.2+open*.65),0,0,Math.PI*2);ctx.fill();
      ctx.fillStyle='#df8494';ctx.beginPath();ctx.ellipse(0,h*open*.25,w*.22,h*open*.3,0,0,Math.PI);ctx.fill();ctx.restore();
    }
    _faceOverlay(layer,p,expression) {
      if(!expression.blush&&!expression.tear)return;
      const b=layer.bounds,w=b[2]-b[0],h=b[3]-b[1],ctx=this.ctx;
      for(const side of [-1,1]) {
        const center=this._headPoint((b[0]+b[2])/2+side*w*.23,b[1]+h*.62,p);
        if(expression.blush>0) {
          ctx.save();ctx.translate(...center);ctx.rotate(p.angleZ*.12*this.strength);
          const gradient=ctx.createRadialGradient(0,0,0,0,0,w*.12);
          gradient.addColorStop(0,`rgba(240,101,128,${expression.blush*.38})`);gradient.addColorStop(1,'rgba(240,101,128,0)');
          ctx.scale(1,.5);ctx.fillStyle=gradient;ctx.beginPath();ctx.arc(0,0,w*.12,0,Math.PI*2);ctx.fill();ctx.restore();
        }
        if(expression.tear>0) {
          ctx.save();ctx.translate(center[0],center[1]-h*.06);ctx.rotate(p.angleZ*.12*this.strength);
          ctx.fillStyle='rgba(147,209,243,.7)';ctx.beginPath();ctx.ellipse(0,h*.08,w*.025,h*.1,0,0,Math.PI*2);ctx.fill();ctx.restore();
        }
      }
    }
    render(timestamp=0) {
      if(!this.loaded||this.destroyed)return;
      if(typeof timestamp!=='number'||!Number.isFinite(timestamp))throw new Error('描画時刻が不正です。');
      const seconds=timestamp/1000,p=this._frameParameters(seconds),expression=this._expression();
      const ctx=this.ctx,canvas=this.canvas;ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.globalAlpha=1;ctx.globalCompositeOperation='source-over';
      const scale=Math.min(canvas.width/this.width,canvas.height/this.height);
      ctx.setTransform(scale,0,0,scale,(canvas.width-this.width*scale)/2,(canvas.height-this.height*scale)/2);
      ctx.imageSmoothingEnabled=true;
      const neutral=!this.idle&&!this.autoBlink&&this.motion==='idle'&&this.expression==='neutral'&&
        !this.pointer.active&&p.angleX===0&&p.angleY===0&&p.angleZ===0&&p.eyeOpen===1&&p.mouthOpen===0&&p.breath===0;
      for(const layer of this.layers) {
        if(layer.visible===false)continue;
        const b=layer.bounds,w=b[2]-b[0],h=b[3]-b[1];
        if(neutral){ctx.drawImage(layer.image,b[0],b[1],w,h);continue;}
        if(layer.role==='mouth')this._mouthInterior(layer,p,expression);
        const n=this.resolution,points=[];
        for(let yy=0;yy<=n;yy++)for(let xx=0;xx<=n;xx++) {
          const sx=w*xx/n,sy=h*yy/n;
          points.push({src:[sx,sy],dst:this._vertex(layer,b[0]+sx,b[1]+sy,p,expression,seconds)});
        }
        // Draw any affine deformation once. Independently clipped triangles
        // antialias their shared edges and otherwise change translucent artwork
        // even for identity/head rotations/eye compression/static layers.
        const origin=points[0].dst,right=points[n].dst,bottom=points[n*(n+1)].dst;
        const a=(right[0]-origin[0])/w,b2=(right[1]-origin[1])/w;
        const c=(bottom[0]-origin[0])/h,d2=(bottom[1]-origin[1])/h;
        const affine=points.every(v=>Math.abs(v.dst[0]-origin[0]-a*v.src[0]-c*v.src[1])<1e-7&&
                                    Math.abs(v.dst[1]-origin[1]-b2*v.src[0]-d2*v.src[1])<1e-7);
        if(affine) {
          ctx.save();ctx.transform(a,b2,c,d2,origin[0],origin[1]);ctx.drawImage(layer.image,0,0);ctx.restore();
          if(layer.role==='face')this._faceOverlay(layer,p,expression);
          continue;
        }
        // Assemble a layer separately with additive premultiplied coverage.
        // Shared clip edges have complementary antialias coverage. Summing it
        // preserves the texture's alpha; source-over would leave a grid, while
        // expanding clips would accumulate translucent pixels along every edge.
        const left=Math.floor(Math.min(...points.map(v=>v.dst[0])))-1;
        const top=Math.floor(Math.min(...points.map(v=>v.dst[1])))-1;
        const meshWidth=Math.ceil(Math.max(...points.map(v=>v.dst[0])))-left+1;
        const meshHeight=Math.ceil(Math.max(...points.map(v=>v.dst[1])))-top+1;
        const meshCanvas=this._meshCanvas,mesh=this._meshContext;
        if(meshCanvas.width<meshWidth)meshCanvas.width=meshWidth;
        if(meshCanvas.height<meshHeight)meshCanvas.height=meshHeight;
        mesh.setTransform(1,0,0,1,0,0);mesh.clearRect(0,0,meshCanvas.width,meshCanvas.height);
        mesh.globalCompositeOperation='lighter';mesh.imageSmoothingEnabled=true;mesh.translate(-left,-top);
        for(let yy=0;yy<n;yy++)for(let xx=0;xx<n;xx++) {
          const a=points[yy*(n+1)+xx],b1=points[yy*(n+1)+xx+1],c=points[(yy+1)*(n+1)+xx],d=points[(yy+1)*(n+1)+xx+1];
          triangle(mesh,layer.image,[a.src,b1.src,d.src],[a.dst,b1.dst,d.dst]);
          triangle(mesh,layer.image,[a.src,d.src,c.src],[a.dst,d.dst,c.dst]);
        }
        ctx.drawImage(meshCanvas,0,0,meshWidth,meshHeight,left,top,meshWidth,meshHeight);
        if(layer.role==='face')this._faceOverlay(layer,p,expression);
      }
      ctx.setTransform(1,0,0,1,0,0);
    }
  }
  global.Live2DWeb=Object.freeze({Character,expressionNames:Object.freeze(expressionNames),motionNames:Object.freeze(motionNames),version:1});
})(typeof window!=='undefined'?window:globalThis);
