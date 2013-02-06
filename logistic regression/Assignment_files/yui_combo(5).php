/*
YUI 3.5.1 (build 22)
Copyright 2012 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/
YUI.add("attribute-core",function(c){c.State=function(){this.data={};};c.State.prototype={add:function(v,w,y){var x=this.data;x[v]=x[v]||{};x[v][w]=y;},addAll:function(v,x){var w;for(w in x){if(x.hasOwnProperty(w)){this.add(v,w,x[w]);}}},remove:function(v,w){var x=this.data;if(x[v]){delete x[v][w];}},removeAll:function(v,x){var w=this.data;if(!x){if(w[v]){delete w[v];}}else{c.each(x,function(z,y){if(c.Lang.isString(y)){this.remove(v,y);}else{this.remove(v,z);}},this);}},get:function(v,w){var x=this.data;return(x[v])?x[v][w]:undefined;},getAll:function(w,v){var y=this.data,x;if(!v){c.each(y[w],function(A,z){x=x||{};x[z]=A;});}else{x=y[w];}return x;}};var i=c.Object,d=c.Lang,q=".",k="getter",j="setter",l="readOnly",r="writeOnce",p="initOnly",u="validator",f="value",m="valueFn",o="lazyAdd",t="added",h="_bypassProxy",b="initializing",g="initValue",a="lazy",n="isLazyAdd",e;function s(w,v,x){this._initAttrHost(w,v,x);}s.INVALID_VALUE={};e=s.INVALID_VALUE;s._ATTR_CFG=[j,k,u,f,m,r,l,o,h];s.prototype={_initAttrHost:function(w,v,x){this._state=new c.State();this._initAttrs(w,v,x);},addAttr:function(w,v,y){var z=this,B=z._state,A,x;v=v||{};y=(o in v)?v[o]:y;if(y&&!z.attrAdded(w)){B.addAll(w,{lazy:v,added:true});}else{if(!z.attrAdded(w)||B.get(w,n)){x=(f in v);if(x){A=v.value;delete v.value;}v.added=true;v.initializing=true;B.addAll(w,v);if(x){z.set(w,A);}B.remove(w,b);}}return z;},attrAdded:function(v){return !!this._state.get(v,t);},get:function(v){return this._getAttr(v);},_isLazyAttr:function(v){return this._state.get(v,a);},_addLazyAttr:function(x,v){var y=this._state,w=y.get(x,a);y.add(x,n,true);y.remove(x,a);this.addAttr(x,w);},set:function(v,w){return this._setAttr(v,w);},_set:function(v,w){return this._setAttr(v,w,null,true);},_setAttr:function(x,A,v,y){var E=true,w=this._state,B=this._stateProxy,H,D,G,I,z,C,F;if(x.indexOf(q)!==-1){G=x;I=x.split(q);x=I.shift();}if(this._isLazyAttr(x)){this._addLazyAttr(x);}H=w.getAll(x,true)||{};D=(!(f in H));if(B&&x in B&&!H._bypassProxy){D=false;}C=H.writeOnce;F=H.initializing;if(!D&&!y){if(C){E=false;}if(H.readOnly){E=false;}}if(!F&&!y&&C===p){E=false;}if(E){if(!D){z=this.get(x);}if(I){A=i.setValue(c.clone(z),I,A);if(A===undefined){E=false;}}if(E){if(!this._fireAttrChange||F){this._setAttrVal(x,G,z,A);}else{this._fireAttrChange(x,G,z,A,v);}}}return this;},_getAttr:function(x){var y=this,C=x,z=y._state,A,v,B,w;if(x.indexOf(q)!==-1){A=x.split(q);x=A.shift();}if(y._tCfgs&&y._tCfgs[x]){w={};w[x]=y._tCfgs[x];delete y._tCfgs[x];y._addAttrs(w,y._tVals);}if(y._isLazyAttr(x)){y._addLazyAttr(x);}B=y._getStateVal(x);v=z.get(x,k);if(v&&!v.call){v=this[v];}B=(v)?v.call(y,B,C):B;B=(A)?i.getValue(B,A):B;return B;},_getStateVal:function(v){var w=this._stateProxy;return w&&(v in w)&&!this._state.get(v,h)?w[v]:this._state.get(v,f);},_setStateVal:function(v,x){var w=this._stateProxy;if(w&&(v in w)&&!this._state.get(v,h)){w[v]=x;}else{this._state.add(v,f,x);}},_setAttrVal:function(H,G,C,A){var I=this,D=true,F=this._state.getAll(H,true)||{},y=F.validator,B=F.setter,E=F.initializing,x=this._getStateVal(H),w=G||H,z,v;if(y){if(!y.call){y=this[y];}if(y){v=y.call(I,A,w);if(!v&&E){A=F.defaultValue;v=true;}}}if(!y||v){if(B){if(!B.call){B=this[B];}if(B){z=B.call(I,A,w);if(z===e){D=false;}else{if(z!==undefined){A=z;}}}}if(D){if(!G&&(A===x)&&!d.isObject(A)){D=false;}else{if(!(g in F)){F.initValue=A;}I._setStateVal(H,A);}}}else{D=false;}return D;},setAttrs:function(v){return this._setAttrs(v);},_setAttrs:function(w){for(var v in w){if(w.hasOwnProperty(v)){this.set(v,w[v]);}}return this;},getAttrs:function(v){return this._getAttrs(v);},_getAttrs:function(y){var A=this,C={},z,w,v,B,x=(y===true);y=(y&&!x)?y:i.keys(A._state.data);for(z=0,w=y.length;z<w;z++){v=y[z];B=A.get(v);if(!x||A._getStateVal(v)!=A._state.get(v,g)){C[v]=A.get(v);}}return C;},addAttrs:function(v,w,x){var y=this;if(v){y._tCfgs=v;y._tVals=y._normAttrVals(w);y._addAttrs(v,y._tVals,x);y._tCfgs=y._tVals=null;}return y;},_addAttrs:function(w,x,y){var A=this,v,z,B;for(v in w){if(w.hasOwnProperty(v)){z=w[v];z.defaultValue=z.value;B=A._getAttrInitVal(v,z,A._tVals);if(B!==undefined){z.value=B;}if(A._tCfgs[v]){delete A._tCfgs[v];}A.addAttr(v,z,y);}}},_protectAttrs:function(w){if(w){w=c.merge(w);for(var v in w){if(w.hasOwnProperty(v)){w[v]=c.merge(w[v]);}}}return w;},_normAttrVals:function(v){return(v)?c.merge(v):null;},_getAttrInitVal:function(v,w,y){var z,x;if(!w.readOnly&&y&&y.hasOwnProperty(v)){z=y[v];}else{z=w.value;x=w.valueFn;if(x){if(!x.call){x=this[x];}if(x){z=x.call(this,v);}}}return z;},_initAttrs:function(w,v,z){w=w||this.constructor.ATTRS;var y=c.Base,x=c.BaseCore,A=(y&&c.instanceOf(this,y)),B=(!A&&x&&c.instanceOf(this,x));if(w&&!A&&!B){this.addAttrs(this._protectAttrs(w),v,z);}}};c.AttributeCore=s;},"3.5.1");/*
YUI 3.5.1 (build 22)
Copyright 2012 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/
YUI.add("base-core",function(a){var e=a.Object,i=a.Lang,h=".",l="initialized",d="destroyed",c="initializer",b=Object.prototype.constructor,j="deep",m="shallow",k="destructor",g=a.AttributeCore,f=function(t,q,o){var u;for(u in q){if(o[u]){t[u]=q[u];}}return t;};function n(o){if(!this._BaseInvoked){this._BaseInvoked=true;this._initBase(o);}}n._ATTR_CFG=g._ATTR_CFG.concat("cloneDefaultValue");n._ATTR_CFG_HASH=a.Array.hash(n._ATTR_CFG);n._NON_ATTRS_CFG=["plugins"];n.NAME="baseCore";n.ATTRS={initialized:{readOnly:true,value:false},destroyed:{readOnly:true,value:false}};n.prototype={_initBase:function(o){a.stamp(this);this._initAttribute(o);var p=a.Plugin&&a.Plugin.Host;if(this._initPlugins&&p){p.call(this);}if(this._lazyAddAttrs!==false){this._lazyAddAttrs=true;}this.name=this.constructor.NAME;this.init.apply(this,arguments);},_initAttribute:function(){g.apply(this);},init:function(o){this._baseInit(o);return this;},_baseInit:function(o){this._initHierarchy(o);if(this._initPlugins){this._initPlugins(o);}this._set(l,true);},destroy:function(){this._baseDestroy();return this;},_baseDestroy:function(){if(this._destroyPlugins){this._destroyPlugins();}this._destroyHierarchy();this._set(d,true);},_getClasses:function(){if(!this._classes){this._initHierarchyData();}return this._classes;},_getAttrCfgs:function(){if(!this._attrs){this._initHierarchyData();}return this._attrs;},_filterAttrCfgs:function(s,p){var q=null,o,r=s.ATTRS;if(r){for(o in r){if(p[o]){q=q||{};q[o]=p[o];p[o]=null;}}}return q;},_filterAdHocAttrs:function(r,p){var q,s=this._nonAttrs,o;if(p){q={};for(o in p){if(!r[o]&&!s[o]&&p.hasOwnProperty(o)){q[o]={value:p[o]};}}}return q;},_initHierarchyData:function(){var u=this.constructor,r,o,s,t=(this._allowAdHocAttrs)?{}:null,q=[],p=[];while(u){q[q.length]=u;if(u.ATTRS){p[p.length]=u.ATTRS;}if(this._allowAdHocAttrs){s=u._NON_ATTRS_CFG;if(s){for(r=0,o=s.length;r<o;r++){t[s[r]]=true;}}}u=u.superclass?u.superclass.constructor:null;}this._classes=q;this._nonAttrs=t;this._attrs=this._aggregateAttrs(p);},_attrCfgHash:function(){return n._ATTR_CFG_HASH;},_aggregateAttrs:function(v){var r,w,q,o,x,p,u,t=this._attrCfgHash(),s={};if(v){for(p=v.length-1;p>=0;--p){w=v[p];for(r in w){if(w.hasOwnProperty(r)){q=f({},w[r],t);o=q.value;u=q.cloneDefaultValue;if(o){if((u===undefined&&(b===o.constructor||i.isArray(o)))||u===j||u===true){q.value=a.clone(o);}else{if(u===m){q.value=a.merge(o);}}}x=null;if(r.indexOf(h)!==-1){x=r.split(h);r=x.shift();}if(x&&s[r]&&s[r].value){e.setValue(s[r].value,x,o);}else{if(!x){if(!s[r]){s[r]=q;}else{f(s[r],q,t);}}}}}}}return s;},_initHierarchy:function(u){var q=this._lazyAddAttrs,v,x,z,s,p,y,t,r=this._getClasses(),o=this._getAttrCfgs(),w=r.length-1;for(z=w;z>=0;z--){v=r[z];x=v.prototype;t=v._yuibuild&&v._yuibuild.exts;if(t){for(s=0,p=t.length;s<p;s++){t[s].apply(this,arguments);}}this.addAttrs(this._filterAttrCfgs(v,o),u,q);if(this._allowAdHocAttrs&&z===w){this.addAttrs(this._filterAdHocAttrs(o,u),u,q);}if(x.hasOwnProperty(c)){x.initializer.apply(this,arguments);}if(t){for(s=0;s<p;s++){y=t[s].prototype;if(y.hasOwnProperty(c)){y.initializer.apply(this,arguments);}}}}},_destroyHierarchy:function(){var s,t,w,u,q,o,r,v,p=this._getClasses();for(w=0,u=p.length;w<u;w++){s=p[w];t=s.prototype;r=s._yuibuild&&s._yuibuild.exts;if(r){for(q=0,o=r.length;q<o;q++){v=r[q].prototype;if(v.hasOwnProperty(k)){v.destructor.apply(this,arguments);}}}if(t.hasOwnProperty(k)){t.destructor.apply(this,arguments);}}},toString:function(){return this.name+"["+a.stamp(this,true)+"]";}};a.mix(n,g,false,null,1);n.prototype.constructor=n;a.BaseCore=n;},"3.5.1",{requires:["attribute-core"]});/*
YUI 3.5.1 (build 22)
Copyright 2012 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/
YUI.add("event-custom-complex",function(f){var b,e,d={},a=f.CustomEvent.prototype,c=f.EventTarget.prototype;f.EventFacade=function(h,g){h=h||d;this._event=h;this.details=h.details;this.type=h.type;this._type=h.type;this.target=h.target;this.currentTarget=g;this.relatedTarget=h.relatedTarget;};f.extend(f.EventFacade,Object,{stopPropagation:function(){this._event.stopPropagation();this.stopped=1;},stopImmediatePropagation:function(){this._event.stopImmediatePropagation();this.stopped=2;},preventDefault:function(){this._event.preventDefault();this.prevented=1;},halt:function(g){this._event.halt(g);this.prevented=1;this.stopped=(g)?2:1;}});a.fireComplex=function(p){var r,l,g,n,i,o,u,j,h,t=this,s=t.host||t,m,k;if(t.stack){if(t.queuable&&t.type!=t.stack.next.type){t.log("queue "+t.type);t.stack.queue.push([t,p]);return true;}}r=t.stack||{id:t.id,next:t,silent:t.silent,stopped:0,prevented:0,bubbling:null,type:t.type,afterQueue:new f.Queue(),defaultTargetOnly:t.defaultTargetOnly,queue:[]};j=t.getSubs();t.stopped=(t.type!==r.type)?0:r.stopped;t.prevented=(t.type!==r.type)?0:r.prevented;t.target=t.target||s;u=new f.EventTarget({fireOnce:true,context:s});t.events=u;if(t.stoppedFn){u.on("stopped",t.stoppedFn);}t.currentTarget=s;t.details=p.slice();t.log("Firing "+t.type);t._facade=null;l=t._getFacade(p);if(f.Lang.isObject(p[0])){p[0]=l;}else{p.unshift(l);}if(j[0]){t._procSubs(j[0],p,l);}if(t.bubbles&&s.bubble&&!t.stopped){k=r.bubbling;r.bubbling=t.type;if(r.type!=t.type){r.stopped=0;r.prevented=0;}o=s.bubble(t,p,null,r);t.stopped=Math.max(t.stopped,r.stopped);t.prevented=Math.max(t.prevented,r.prevented);r.bubbling=k;}if(t.prevented){if(t.preventedFn){t.preventedFn.apply(s,p);}}else{if(t.defaultFn&&((!t.defaultTargetOnly&&!r.defaultTargetOnly)||s===l.target)){t.defaultFn.apply(s,p);}}t._broadcast(p);if(j[1]&&!t.prevented&&t.stopped<2){if(r.id===t.id||t.type!=s._yuievt.bubbling){t._procSubs(j[1],p,l);while((m=r.afterQueue.last())){m();}}else{h=j[1];if(r.execDefaultCnt){h=f.merge(h);f.each(h,function(q){q.postponed=true;});}r.afterQueue.add(function(){t._procSubs(h,p,l);});}}t.target=null;if(r.id===t.id){n=r.queue;while(n.length){g=n.pop();i=g[0];r.next=i;i.fire.apply(i,g[1]);}t.stack=null;}o=!(t.stopped);if(t.type!=s._yuievt.bubbling){r.stopped=0;r.prevented=0;t.stopped=0;t.prevented=0;}return o;};a._getFacade=function(){var g=this._facade,j,i,h=this.details;if(!g){g=new f.EventFacade(this,this.currentTarget);}j=h&&h[0];if(f.Lang.isObject(j,true)){i={};f.mix(i,g,true,e);f.mix(g,j,true);f.mix(g,i,true,e);g.type=j.type||g.type;}g.details=this.details;g.target=this.originalTarget||this.target;g.currentTarget=this.currentTarget;g.stopped=0;g.prevented=0;this._facade=g;return this._facade;};a.stopPropagation=function(){this.stopped=1;if(this.stack){this.stack.stopped=1;}this.events.fire("stopped",this);};a.stopImmediatePropagation=function(){this.stopped=2;if(this.stack){this.stack.stopped=2;}this.events.fire("stopped",this);};a.preventDefault=function(){if(this.preventable){this.prevented=1;if(this.stack){this.stack.prevented=1;}}};a.halt=function(g){if(g){this.stopImmediatePropagation();}else{this.stopPropagation();}this.preventDefault();};c.addTarget=function(g){this._yuievt.targets[f.stamp(g)]=g;this._yuievt.hasTargets=true;};c.getTargets=function(){return f.Object.values(this._yuievt.targets);};c.removeTarget=function(g){delete this._yuievt.targets[f.stamp(g)];};c.bubble=function(u,q,o,s){var m=this._yuievt.targets,p=true,v,r=u&&u.type,h,l,n,j,g=o||(u&&u.target)||this,k;if(!u||((!u.stopped)&&m)){for(l in m){if(m.hasOwnProperty(l)){v=m[l];h=v.getEvent(r,true);j=v.getSibling(r,h);if(j&&!h){h=v.publish(r);}k=v._yuievt.bubbling;v._yuievt.bubbling=r;if(!h){if(v._yuievt.hasTargets){v.bubble(u,q,g,s);}}else{h.sibling=j;h.target=g;h.originalTarget=g;h.currentTarget=v;n=h.broadcast;h.broadcast=false;h.emitFacade=true;h.stack=s;p=p&&h.fire.apply(h,q||u.details||[]);h.broadcast=n;h.originalTarget=null;if(h.stopped){break;}}v._yuievt.bubbling=k;}}}return p;};b=new f.EventFacade();e=f.Object.keys(b);},"3.5.1",{requires:["event-custom-base"]});/*
YUI 3.5.1 (build 22)
Copyright 2012 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/
YUI.add("attribute-events",function(e){var f=e.EventTarget,d="Change",a="broadcast",c="published";function b(){this._ATTR_E_FACADE={};f.call(this,{emitFacade:true});}b._ATTR_CFG=[a];b.prototype={set:function(g,i,h){return this._setAttr(g,i,h);},_set:function(g,i,h){return this._setAttr(g,i,h,true);},setAttrs:function(g,h){return this._setAttrs(g,h);},_fireAttrChange:function(o,n,k,j,g){var q=this,m=o+d,i=q._state,p,l,h;if(!i.get(o,c)){h={queuable:false,defaultTargetOnly:true,defaultFn:q._defAttrChangeFn,silent:true};l=i.get(o,a);if(l!==undefined){h.broadcast=l;}q.publish(m,h);i.add(o,c,true);}p=(g)?e.merge(g):q._ATTR_E_FACADE;p.attrName=o;p.subAttrName=n;p.prevVal=k;p.newVal=j;q.fire(m,p);},_defAttrChangeFn:function(g){if(!this._setAttrVal(g.attrName,g.subAttrName,g.prevVal,g.newVal)){g.stopImmediatePropagation();}else{g.newVal=this.get(g.attrName);}}};e.mix(b,f,false,null,1);e.AttributeEvents=b;},"3.5.1",{requires:["event-custom"]});/*
YUI 3.5.1 (build 22)
Copyright 2012 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/
YUI.add("attribute-extras",function(f){var a="broadcast",d="published",e="initValue",c={readOnly:1,writeOnce:1,getter:1,broadcast:1};function b(){}b.prototype={modifyAttr:function(h,g){var i=this,k,j;if(i.attrAdded(h)){if(i._isLazyAttr(h)){i._addLazyAttr(h);}j=i._state;for(k in g){if(c[k]&&g.hasOwnProperty(k)){j.add(h,k,g[k]);if(k===a){j.remove(h,d);}}}}},removeAttr:function(g){this._state.removeAll(g);},reset:function(g){var h=this;if(g){if(h._isLazyAttr(g)){h._addLazyAttr(g);}h.set(g,h._state.get(g,e));}else{f.each(h._state.data,function(i,j){h.reset(j);});}return h;},_getAttrCfg:function(g){var i,h=this._state;if(g){i=h.getAll(g)||{};}else{i={};f.each(h.data,function(j,k){i[k]=h.getAll(k);});}return i;}};f.AttributeExtras=b;},"3.5.1");/*
YUI 3.5.1 (build 22)
Copyright 2012 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/
YUI.add("attribute-base",function(b){var a=function(){this._ATTR_E_FACADE=null;this._yuievt=null;b.AttributeCore.apply(this,arguments);b.AttributeEvents.apply(this,arguments);b.AttributeExtras.apply(this,arguments);};b.mix(a,b.AttributeCore,false,null,1);b.mix(a,b.AttributeExtras,false,null,1);b.mix(a,b.AttributeEvents,true,null,1);a.INVALID_VALUE=b.AttributeCore.INVALID_VALUE;a._ATTR_CFG=b.AttributeCore._ATTR_CFG.concat(b.AttributeEvents._ATTR_CFG);b.Attribute=a;},"3.5.1",{requires:["attribute-core","attribute-events","attribute-extras"]});/*
YUI 3.5.1 (build 22)
Copyright 2012 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/
YUI.add("base-base",function(b){var g=b.Lang,e="destroy",i="init",h="bubbleTargets",c="_bubbleTargets",j=b.BaseCore,f=b.AttributeCore,a=b.Attribute;function d(){j.apply(this,arguments);}d._ATTR_CFG=a._ATTR_CFG.concat("cloneDefaultValue");d._ATTR_CFG_HASH=b.Array.hash(d._ATTR_CFG);d._NON_ATTRS_CFG=j._NON_ATTRS_CFG.concat(["on","after","bubbleTargets"]);d.NAME="base";d.ATTRS=f.prototype._protectAttrs(j.ATTRS);d.prototype={_initBase:function(k){this._eventPrefix=this.constructor.EVENT_PREFIX||this.constructor.NAME;b.BaseCore.prototype._initBase.call(this,k);},_initAttribute:function(k){a.call(this);this._yuievt.config.prefix=this._eventPrefix;},_attrCfgHash:function(){return d._ATTR_CFG_HASH;},init:function(k){this.publish(i,{queuable:false,fireOnce:true,defaultTargetOnly:true,defaultFn:this._defInitFn});this._preInitEventCfg(k);this.fire(i,{cfg:k});return this;},_preInitEventCfg:function(m){if(m){if(m.on){this.on(m.on);}if(m.after){this.after(m.after);}}var n,k,p,o=(m&&h in m);if(o||c in this){p=o?(m&&m.bubbleTargets):this._bubbleTargets;if(g.isArray(p)){for(n=0,k=p.length;n<k;n++){this.addTarget(p[n]);}}else{if(p){this.addTarget(p);}}}},destroy:function(){this.publish(e,{queuable:false,fireOnce:true,defaultTargetOnly:true,defaultFn:this._defDestroyFn});this.fire(e);this.detachAll();return this;},_defInitFn:function(k){this._baseInit(k.cfg);},_defDestroyFn:function(k){this._baseDestroy(k.cfg);}};b.mix(d,a,false,null,1);b.mix(d,j,false,null,1);d.prototype.constructor=d;b.Base=d;},"3.5.1",{requires:["base-core","attribute-base"]});/*
YUI 3.5.1 (build 22)
Copyright 2012 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/
YUI.add("base-pluginhost",function(c){var a=c.Base,b=c.Plugin.Host;c.mix(a,b,false,null,1);a.plug=b.plug;a.unplug=b.unplug;},"3.5.1",{requires:["base-base","pluginhost"]});/*
YUI 3.5.1 (build 22)
Copyright 2012 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/
YUI.add("base-build",function(g){var c=g.Base,a=g.Lang,b="initializer",e="destructor",f,d=function(j,i,h){if(h[j]){i[j]=(i[j]||[]).concat(h[j]);}};c._build=function(h,o,t,x,w,q){var y=c._build,j=y._ctor(o,q),m=y._cfg(o,q,t),v=y._mixCust,k=j._yuibuild.dynamic,p,n,u,z,s,r;for(p=0,n=t.length;p<n;p++){u=t[p];z=u.prototype;s=z[b];r=z[e];delete z[b];delete z[e];g.mix(j,u,true,null,1);v(j,u,m);if(s){z[b]=s;}if(r){z[e]=r;}j._yuibuild.exts.push(u);}if(x){g.mix(j.prototype,x,true);}if(w){g.mix(j,y._clean(w,m),true);v(j,w,m);}j.prototype.hasImpl=y._impl;if(k){j.NAME=h;j.prototype.constructor=j;}return j;};f=c._build;g.mix(f,{_mixCust:function(h,t,p){var o,j,q,k,m,n;if(p){o=p.aggregates;j=p.custom;q=p.statics;}if(q){g.mix(h,t,true,q);}if(o){for(n=0,m=o.length;n<m;n++){k=o[n];if(!h.hasOwnProperty(k)&&t.hasOwnProperty(k)){h[k]=a.isArray(t[k])?[]:{};}g.aggregate(h,t,true,[k]);}}if(j){for(n in j){if(j.hasOwnProperty(n)){j[n](n,h,t);}}}},_tmpl:function(h){function i(){i.superclass.constructor.apply(this,arguments);}g.extend(i,h);return i;},_impl:function(n){var q=this._getClasses(),p,k,h,o,r,m;for(p=0,k=q.length;p<k;p++){h=q[p];if(h._yuibuild){o=h._yuibuild.exts;r=o.length;for(m=0;m<r;m++){if(o[m]===n){return true;}}}}return false;},_ctor:function(h,i){var k=(i&&false===i.dynamic)?false:true,l=(k)?f._tmpl(h):h,j=l._yuibuild;if(!j){j=l._yuibuild={};}j.id=j.id||null;j.exts=j.exts||[];j.dynamic=k;return l;},_cfg:function(m,q,n){var k=[],p={},v=[],h,t=(q&&q.aggregates),u=(q&&q.custom),r=(q&&q.statics),s=m,o,j;while(s&&s.prototype){h=s._buildCfg;if(h){if(h.aggregates){k=k.concat(h.aggregates);}if(h.custom){g.mix(p,h.custom,true);}if(h.statics){v=v.concat(h.statics);}}s=s.superclass?s.superclass.constructor:null;}if(n){for(o=0,j=n.length;o<j;o++){s=n[o];h=s._buildCfg;if(h){if(h.aggregates){k=k.concat(h.aggregates);}if(h.custom){g.mix(p,h.custom,true);}if(h.statics){v=v.concat(h.statics);}}}}if(t){k=k.concat(t);}if(u){g.mix(p,q.cfgBuild,true);}if(r){v=v.concat(r);}return{aggregates:k,custom:p,statics:v};},_clean:function(q,j){var p,k,h,n=g.merge(q),o=j.aggregates,m=j.custom;for(p in m){if(n.hasOwnProperty(p)){delete n[p];}}for(k=0,h=o.length;k<h;k++){p=o[k];if(n.hasOwnProperty(p)){delete n[p];}}return n;}});c.build=function(j,h,k,i){return f(j,h,k,null,null,i);};c.create=function(h,k,j,i,l){return f(h,k,j,i,l);};c.mix=function(h,i){return f(null,h,i,null,null,{dynamic:false});};c._buildCfg={custom:{ATTRS:function(m,k,i){k.ATTRS=k.ATTRS||{};if(i.ATTRS){var j=i.ATTRS,l=k.ATTRS,h;for(h in j){if(j.hasOwnProperty(h)){l[h]=l[h]||{};g.mix(l[h],j[h],true);}}}},_NON_ATTRS_CFG:d},aggregates:["_PLUG","_UNPLUG"]};},"3.5.1",{requires:["base-base"]});/*
YUI 3.5.1 (build 22)
Copyright 2012 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/
YUI.add("event-synthetic",function(b){var j=b.Env.evt.dom_map,d=b.Array,i=b.Lang,l=i.isObject,c=i.isString,e=i.isArray,g=b.Selector.query,k=function(){};function h(n,m){this.handle=n;this.emitFacade=m;}h.prototype.fire=function(s){var t=d(arguments,0,true),q=this.handle,o=q.evt,m=q.sub,p=m.context,u=m.filter,n=s||{},r;if(this.emitFacade){if(!s||!s.preventDefault){n=o._getFacade();if(l(s)&&!s.preventDefault){b.mix(n,s,true);t[0]=n;}else{t.unshift(n);}}n.type=o.type;n.details=t.slice();if(u){n.container=o.host;}}else{if(u&&l(s)&&s.currentTarget){t.shift();}}m.context=p||n.currentTarget||o.host;r=o.fire.apply(o,t);m.context=p;return r;};function f(o,n,m){this.handles=[];this.el=o;this.key=m;this.domkey=n;}f.prototype={constructor:f,type:"_synth",fn:k,capture:false,register:function(m){m.evt.registry=this;this.handles.push(m);},unregister:function(p){var o=this.handles,n=j[this.domkey],m;for(m=o.length-1;m>=0;--m){if(o[m].sub===p){o.splice(m,1);break;}}if(!o.length){delete n[this.key];if(!b.Object.size(n)){delete j[this.domkey];}}},detachAll:function(){var n=this.handles,m=n.length;while(--m>=0){n[m].detach();}}};function a(){this._init.apply(this,arguments);}b.mix(a,{Notifier:h,SynthRegistry:f,getRegistry:function(s,r,p){var q=s._node,o=b.stamp(q),n="event:"+o+r+"_synth",m=j[o];if(p){if(!m){m=j[o]={};}if(!m[n]){m[n]=new f(q,o,n);}}return(m&&m[n])||null;},_deleteSub:function(n){if(n&&n.fn){var m=this.eventDef,o=(n.filter)?"detachDelegate":"detach";this.subscribers={};this.subCount=0;m[o](n.node,n,this.notifier,n.filter);this.registry.unregister(n);delete n.fn;delete n.node;delete n.context;}},prototype:{constructor:a,_init:function(){var m=this.publishConfig||(this.publishConfig={});this.emitFacade=("emitFacade" in m)?m.emitFacade:true;m.emitFacade=false;},processArgs:k,on:k,detach:k,delegate:k,detachDelegate:k,_on:function(s,t){var u=[],o=s.slice(),p=this.processArgs(s,t),q=s[2],m=t?"delegate":"on",n,r;n=(c(q))?g(q):d(q||b.one(b.config.win));if(!n.length&&c(q)){r=b.on("available",function(){b.mix(r,b[m].apply(b,o),true);},q);return r;}b.Array.each(n,function(w){var x=s.slice(),v;w=b.one(w);if(w){if(t){v=x.splice(3,1)[0];}x.splice(0,4,x[1],x[3]);if(!this.preventDups||!this.getSubs(w,s,null,true)){u.push(this._subscribe(w,m,x,p,v));}}},this);return(u.length===1)?u[0]:new b.EventHandle(u);},_subscribe:function(q,o,t,r,p){var v=new b.CustomEvent(this.type,this.publishConfig),s=v.on.apply(v,t),u=new h(s,this.emitFacade),n=a.getRegistry(q,this.type,true),m=s.sub;m.node=q;m.filter=p;if(r){this.applyArgExtras(r,m);}b.mix(v,{eventDef:this,notifier:u,host:q,currentTarget:q,target:q,el:q._node,_delete:a._deleteSub},true);s.notifier=u;n.register(s);this[o](q,m,u,p);return s;},applyArgExtras:function(m,n){n._extra=m;},_detach:function(o){var t=o[2],r=(c(t))?g(t):d(t),s,q,m,p,n;o.splice(2,1);for(q=0,m=r.length;q<m;++q){s=b.one(r[q]);if(s){p=this.getSubs(s,o);if(p){for(n=p.length-1;n>=0;--n){p[n].detach();}}}}},getSubs:function(o,u,n,q){var m=a.getRegistry(o,this.type),v=[],t,p,s,r;if(m){t=m.handles;if(!n){n=this.subMatch;}for(p=0,s=t.length;p<s;++p){r=t[p];if(n.call(this,r.sub,u)){if(q){return r;}else{v.push(t[p]);}}}}return v.length&&v;},subMatch:function(n,m){return !m[1]||n.fn===m[1];}}},true);b.SyntheticEvent=a;b.Event.define=function(o,n,q){var p,r,m;if(o&&o.type){p=o;q=n;}else{if(n){p=b.merge({type:o},n);}}if(p){if(q||!b.Node.DOM_EVENTS[p.type]){r=function(){a.apply(this,arguments);};b.extend(r,a,p);m=new r();o=m.type;b.Node.DOM_EVENTS[o]=b.Env.evt.plugins[o]={eventDef:m,on:function(){return m._on(d(arguments));},delegate:function(){return m._on(d(arguments),true);},detach:function(){return m._detach(d(arguments));}};}}else{if(c(o)||e(o)){b.Array.each(d(o),function(s){b.Node.DOM_EVENTS[s]=1;});}}return m;};},"3.5.1",{requires:["node-base","event-custom-complex"]});/*
YUI 3.5.1 (build 22)
Copyright 2012 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/
YUI.add("event-mouseenter",function(f){var b=f.Env.evt.dom_wrappers,d=f.DOM.contains,c=f.Array,e=function(){},a={proxyType:"mouseover",relProperty:"fromElement",_notify:function(k,i,h){var g=this._node,j=k.relatedTarget||k[i];if(g!==j&&!d(g,j)){h.fire(new f.DOMEventFacade(k,g,b["event:"+f.stamp(g)+k.type]));}},on:function(k,i,j){var h=f.Node.getDOMNode(k),g=[this.proxyType,this._notify,h,null,this.relProperty,j];i.handle=f.Event._attach(g,{facade:false});},detach:function(h,g){g.handle.detach();},delegate:function(l,j,k,i){var h=f.Node.getDOMNode(l),g=[this.proxyType,e,h,null,k];j.handle=f.Event._attach(g,{facade:false});j.handle.sub.filter=i;j.handle.sub.relProperty=this.relProperty;j.handle.sub._notify=this._filterNotify;},_filterNotify:function(j,p,g){p=p.slice();if(this.args){p.push.apply(p,this.args);}var h=f.delegate._applyFilter(this.filter,p,g),q=p[0].relatedTarget||p[0][this.relProperty],o,k,m,n,l;if(h){h=c(h);for(k=0,m=h.length&&(!o||!o.stopped);k<m;++k){l=h[0];if(!d(l,q)){if(!o){o=new f.DOMEventFacade(p[0],l,g);o.container=f.one(g.el);}o.currentTarget=f.one(l);n=p[1].fire(o);if(n===false){break;}}}}return n;},detachDelegate:function(h,g){g.handle.detach();}};f.Event.define("mouseenter",a,true);f.Event.define("mouseleave",f.merge(a,{proxyType:"mouseout",relProperty:"toElement"}),true);},"3.5.1",{requires:["event-synthetic"]});/*
YUI 3.5.1 (build 22)
Copyright 2012 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/
YUI.add("event-resize",function(a){a.Event.define("windowresize",{on:(a.UA.gecko&&a.UA.gecko<1.91)?function(d,b,c){b._handle=a.Event.attach("resize",function(f){c.fire(f);});}:function(e,c,d){var b=a.config.windowResizeDelay||100;c._handle=a.Event.attach("resize",function(f){if(c._timer){c._timer.cancel();}c._timer=a.later(b,a,function(){d.fire(f);});});},detach:function(c,b){if(b._timer){b._timer.cancel();}b._handle.detach();}});},"3.5.1",{requires:["event-synthetic"]});YUI.add('moodle-block_navigation-navigation', function(Y){

/**
 * A 'actionkey' Event to help with Y.delegate().
 * The event consists of the left arrow, right arrow, enter and space keys.
 * More keys can be mapped to action meanings.
 * actions: collapse , expand, toggle, enter.
 *
 * This event is delegated to branches in the navigation tree.
 * The on() method to subscribe allows specifying the desired trigger actions as JSON.
 *
 * Todo: This could be centralised, a similar Event is defined in blocks/dock.js
 */
Y.Event.define("actionkey", {
   // Webkit and IE repeat keydown when you hold down arrow keys.
    // Opera links keypress to page scroll; others keydown.
    // Firefox prevents page scroll via preventDefault() on either
    // keydown or keypress.
    _event: (Y.UA.webkit || Y.UA.ie) ? 'keydown' : 'keypress',

    _keys: {
        //arrows
        '37': 'collapse',
        '39': 'expand',
        //(@todo: lrt/rtl/M.core_dock.cfg.orientation decision to assign arrow to meanings)
        '32': 'toggle',
        '13': 'enter'
    },

    _keyHandler: function (e, notifier, args) {
        if (!args.actions) {
            var actObj = {collapse:true, expand:true, toggle:true, enter:true};
        } else {
            var actObj = args.actions;
        }
        if (this._keys[e.keyCode] && actObj[this._keys[e.keyCode]]) {
            e.action = this._keys[e.keyCode];
            notifier.fire(e);
        }
    },

    on: function (node, sub, notifier) {
        // subscribe to _event and ask keyHandler to handle with given args[0] (the desired actions).
        if (sub.args == null) {
            //no actions given
            sub._detacher = node.on(this._event, this._keyHandler,this, notifier, {actions:false});
        } else {
            sub._detacher = node.on(this._event, this._keyHandler,this, notifier, sub.args[0]);
        }
    },

    detach: function (node, sub, notifier) {
        //detach our _detacher handle of the subscription made in on()
        sub._detacher.detach();
    },

    delegate: function (node, sub, notifier, filter) {
        // subscribe to _event and ask keyHandler to handle with given args[0] (the desired actions).
        if (sub.args == null) {
            //no actions given
            sub._delegateDetacher = node.delegate(this._event, this._keyHandler,filter, this, notifier, {actions:false});
        } else {
            sub._delegateDetacher = node.delegate(this._event, this._keyHandler,filter, this, notifier, sub.args[0]);
        }
    },

    detachDelegate: function (node, sub, notifier) {
        sub._delegateDetacher.detach();
    }
});

var EXPANSIONLIMIT_EVERYTHING = 0,
    EXPANSIONLIMIT_COURSE     = 20,
    EXPANSIONLIMIT_SECTION    = 30,
    EXPANSIONLIMIT_ACTIVITY   = 40;


/**
 * Navigation tree class.
 *
 * This class establishes the tree initially, creating expandable branches as
 * required, and delegating the expand/collapse event.
 */
var TREE = function(config) {
    TREE.superclass.constructor.apply(this, arguments);
}
TREE.prototype = {
    /**
     * The tree's ID, normally its block instance id.
     */
    id : null,
    /**
     * Initialise the tree object when its first created.
     */
    initializer : function(config) {
        this.id = config.id;

        var node = Y.one('#inst'+config.id);

        // Can't find the block instance within the page
        if (node === null) {
            return;
        }

        // Delegate event to toggle expansion
        var self = this;
        Y.delegate('click', function(e){self.toggleExpansion(e);}, node.one('.block_tree'), '.tree_item.branch');
        Y.delegate('actionkey', function(e){self.toggleExpansion(e);}, node.one('.block_tree'), '.tree_item.branch');

        // Gather the expandable branches ready for initialisation.
        var expansions = [];
        if (config.expansions) {
            expansions = config.expansions;
        } else if (window['navtreeexpansions'+config.id]) {
            expansions = window['navtreeexpansions'+config.id];
        }
        // Establish each expandable branch as a tree branch.
        for (var i in expansions) {
            new BRANCH({
                tree:this,
                branchobj:expansions[i],
                overrides : {
                    expandable : true,
                    children : [],
                    haschildren : true
                }
            }).wire();
            M.block_navigation.expandablebranchcount++;
        }

        // Call the generic blocks init method to add all the generic stuff
        if (this.get('candock')) {
            this.initialise_block(Y, node);
        }
    },
    /**
     * This is a callback function responsible for expanding and collapsing the
     * branches of the tree. It is delegated to rather than multiple event handles.
     */
    toggleExpansion : function(e) {
        // First check if they managed to click on the li iteslf, then find the closest
        // LI ancestor and use that

        if (e.target.test('a') && (e.keyCode == 0 || e.keyCode == 13)) {
            // A link has been clicked (or keypress is 'enter') don't fire any more events just do the default.
            e.stopPropagation();
            return;
        }

        // Makes sure we can get to the LI containing the branch.
        var target = e.target;
        if (!target.test('li')) {
            target = target.ancestor('li')
        }
        if (!target) {
            return;
        }

        // Toggle expand/collapse providing its not a root level branch.
        if (!target.hasClass('depth_1')) {
            if (e.type == 'actionkey') {
                switch (e.action) {
                    case 'expand' :
                        target.removeClass('collapsed');
                        break;
                    case 'collapse' :
                        target.addClass('collapsed');
                        break;
                    default :
                        target.toggleClass('collapsed');
                }
                e.halt();
            } else {
                target.toggleClass('collapsed');
            }
        }

        // If the accordian feature has been enabled collapse all siblings.
        if (this.get('accordian')) {
            target.siblings('li').each(function(){
                if (this.get('id') !== target.get('id') && !this.hasClass('collapsed')) {
                    this.addClass('collapsed');
                }
            });
        }

        // If this block can dock tell the dock to resize if required and check
        // the width on the dock panel in case it is presently in use.
        if (this.get('candock')) {
            M.core_dock.resize();
            var panel = M.core_dock.getPanel();
            if (panel.visible) {
                panel.correctWidth();
            }
        }
    }
}
// The tree extends the YUI base foundation.
Y.extend(TREE, Y.Base, TREE.prototype, {
    NAME : 'navigation-tree',
    ATTRS : {
        instance : {
            value : null
        },
        candock : {
            validator : Y.Lang.isBool,
            value : false
        },
        accordian : {
            validator : Y.Lang.isBool,
            value : false
        },
        expansionlimit : {
            value : 0,
            setter : function(val) {
                return parseInt(val);
            }
        }
    }
});
if (M.core_dock && M.core_dock.genericblock) {
    Y.augment(TREE, M.core_dock.genericblock);
}

/**
 * The tree branch class.
 * This class is used to manage a tree branch, in particular its ability to load
 * its contents by AJAX.
 */
var BRANCH = function(config) {
    BRANCH.superclass.constructor.apply(this, arguments);
}
BRANCH.prototype = {
    /**
     * The node for this branch (p)
     */
    node : null,
    /**
     * A reference to the ajax load event handlers when created.
     */
    event_ajaxload : null,
    event_ajaxload_actionkey : null,
    /**
     * Initialises the branch when it is first created.
     */
    initializer : function(config) {
        if (config.branchobj !== null) {
            // Construct from the provided xml
            for (var i in config.branchobj) {
                this.set(i, config.branchobj[i]);
            }
            var children = this.get('children');
            this.set('haschildren', (children.length > 0));
        }
        if (config.overrides !== null) {
            // Construct from the provided xml
            for (var i in config.overrides) {
                this.set(i, config.overrides[i]);
            }
        }
        // Get the node for this branch
        this.node = Y.one('#', this.get('id'));
        // Now check whether the branch is not expandable because of the expansionlimit
        var expansionlimit = this.get('tree').get('expansionlimit');
        var type = this.get('type');
        if (expansionlimit != EXPANSIONLIMIT_EVERYTHING &&  type >= expansionlimit && type <= EXPANSIONLIMIT_ACTIVITY) {
            this.set('expandable', false);
            this.set('haschildren', false);
        }
    },
    /**
     * Draws the branch within the tree.
     *
     * This function creates a DOM structure for the branch and then injects
     * it into the navigation tree at the correct point.
     */
    draw : function(element) {

        var isbranch = (this.get('expandable') || this.get('haschildren'));
        var branchli = Y.Node.create('<li></li>');
        var link = this.get('link');
        var branchp = Y.Node.create('<p class="tree_item"></p>').setAttribute('id', this.get('id'));
        if (!link) {
            //add tab focus if not link (so still one focus per menu node).
            // it was suggested to have 2 foci. one for the node and one for the link in MDL-27428.
            branchp.setAttribute('tabindex', '0');
        }
        if (isbranch) {
            branchli.addClass('collapsed').addClass('contains_branch');
            branchp.addClass('branch');
        }

        // Prepare the icon, should be an object representing a pix_icon
        var branchicon = false;
        var icon = this.get('icon');
        if (icon && (!isbranch || this.get('type') == 40)) {
            branchicon = Y.Node.create('<img alt="" />');
            branchicon.setAttribute('src', M.util.image_url(icon.pix, icon.component));
            branchli.addClass('item_with_icon');
            if (icon.alt) {
                branchicon.setAttribute('alt', icon.alt);
            }
            if (icon.title) {
                branchicon.setAttribute('title', icon.title);
            }
            if (icon.classes) {
                for (var i in icon.classes) {
                    branchicon.addClass(icon.classes[i]);
                }
            }
        }

        if (!link) {
            if (branchicon) {
                branchp.appendChild(branchicon);
            }
            branchp.append(this.get('name'));
        } else {
            var branchlink = Y.Node.create('<a title="'+this.get('title')+'" href="'+link+'"></a>');
            if (branchicon) {
                branchlink.appendChild(branchicon);
            }
            branchlink.append(this.get('name'));
            if (this.get('hidden')) {
                branchlink.addClass('dimmed');
            }
            branchp.appendChild(branchlink);
        }

        branchli.appendChild(branchp);
        element.appendChild(branchli);
        this.node = branchp;
        return this;
    },
    /**
     * Attaches required events to the branch structure.
     */
    wire : function() {
        this.node = this.node || Y.one('#'+this.get('id'));
        if (!this.node) {
            return false;
        }
        if (this.get('expandable')) {
            this.event_ajaxload = this.node.on('ajaxload|click', this.ajaxLoad, this);
            this.event_ajaxload_actionkey = this.node.on('actionkey', this.ajaxLoad, this);
        }
        return this;
    },
    /**
     * Gets the UL element that children for this branch should be inserted into.
     */
    getChildrenUL : function() {
        var ul = this.node.next('ul');
        if (!ul) {
            ul = Y.Node.create('<ul></ul>');
            this.node.ancestor().append(ul);
        }
        return ul;
    },
    /**
     * Load the content of the branch via AJAX.
     *
     * This function calls ajaxProcessResponse with the result of the AJAX
     * request made here.
     */
    ajaxLoad : function(e) {
        if (e.type == 'actionkey' && e.action != 'enter') {
            e.halt();
        } else {
            e.stopPropagation();
        }
        if (e.type = 'actionkey' && e.action == 'enter' && e.target.test('A')) {
            this.event_ajaxload_actionkey.detach();
            this.event_ajaxload.detach();
            return true; // no ajaxLoad for enter
        }

        if (this.node.hasClass('loadingbranch')) {
            return true;
        }

        this.node.addClass('loadingbranch');

        var params = {
            elementid : this.get('id'),
            id : this.get('key'),
            type : this.get('type'),
            sesskey : M.cfg.sesskey,
            instance : this.get('tree').get('instance')
        };

        Y.io(M.cfg.wwwroot+'/lib/ajax/getnavbranch.php', {
            method:'POST',
            data:  build_querystring(params),
            on: {
                complete: this.ajaxProcessResponse
            },
            context:this
        });
        return true;
    },
    /**
     * Processes an AJAX request to load the content of this branch through
     * AJAX.
     */
    ajaxProcessResponse : function(tid, outcome) {
        this.node.removeClass('loadingbranch');
        this.event_ajaxload.detach();
        this.event_ajaxload_actionkey.detach();
        try {
            var object = Y.JSON.parse(outcome.responseText);
            if (object.children && object.children.length > 0) {
                var coursecount = 0;
                for (var i in object.children) {
                    if (typeof(object.children[i])=='object') {
                        if (object.children[i].type == 20) {
                            coursecount++;
                        }
                        this.addChild(object.children[i]);
                    }
                }
                if (this.get('type') == 10 && coursecount >= M.block_navigation.courselimit) {
                    this.addViewAllCoursesChild(this);
                }
                this.get('tree').toggleExpansion({target:this.node});
                return true;
            }
        } catch (ex) {
            // If we got here then there was an error parsing the result
        }
        // The branch is empty so class it accordingly
        this.node.replaceClass('branch', 'emptybranch');
        return true;
    },
    /**
     * Turns the branch object passed to the method into a proper branch object
     * and then adds it as a child of this branch.
     */
    addChild : function(branchobj) {
        // Make the new branch into an object
        var branch = new BRANCH({tree:this.get('tree'), branchobj:branchobj});
        if (branch.draw(this.getChildrenUL())) {
            branch.wire();
            var count = 0, i, children = branch.get('children');
            for (i in children) {
                // Add each branch to the tree
                if (children[i].type == 20) {
                    count++;
                }
                if (typeof(children[i])=='object') {
                    branch.addChild(children[i]);
                }
            }
            if (branch.get('type') == 10 && count >= M.block_navigation.courselimit) {
                this.addViewAllCoursesChild(branch);
            }
        }
        return true;
    },

    /**
     * Add a link to view all courses in a category
     */
    addViewAllCoursesChild: function(branch) {
        branch.addChild({
            name : M.str.moodle.viewallcourses,
            title : M.str.moodle.viewallcourses,
            link : M.cfg.wwwroot+'/course/category.php?id='+branch.get('key'),
            haschildren : false,
            icon : {'pix':"i/navigationitem",'component':'moodle'}
        });
    }
}
Y.extend(BRANCH, Y.Base, BRANCH.prototype, {
    NAME : 'navigation-branch',
    ATTRS : {
        tree : {
            validator : Y.Lang.isObject
        },
        name : {
            value : '',
            validator : Y.Lang.isString,
            setter : function(val) {
                return val.replace(/\n/g, '<br />');
            }
        },
        title : {
            value : '',
            validator : Y.Lang.isString
        },
        id : {
            value : '',
            validator : Y.Lang.isString,
            getter : function(val) {
                if (val == '') {
                    val = 'expandable_branch_'+M.block_navigation.expandablebranchcount;
                    M.block_navigation.expandablebranchcount++;
                }
                return val;
            }
        },
        key : {
            value : null
        },
        type : {
            value : null
        },
        link : {
            value : false
        },
        icon : {
            value : false,
            validator : Y.Lang.isObject
        },
        expandable : {
            value : false,
            validator : Y.Lang.isBool
        },
        hidden : {
            value : false,
            validator : Y.Lang.isBool
        },
        haschildren : {
            value : false,
            validator : Y.Lang.isBool
        },
        children : {
            value : [],
            validator : Y.Lang.isArray
        }
    }
});

/**
 * This namespace will contain all of the contents of the navigation blocks
 * global navigation and settings.
 * @namespace
 */
M.block_navigation = M.block_navigation || {
    /** The number of expandable branches in existence */
    expandablebranchcount:1,
    courselimit : 20,
    instance : null,
    /**
     * Add new instance of navigation tree to tree collection
     */
    init_add_tree:function(properties) {
        if (properties.courselimit) {
            this.courselimit = properties.courselimit;
        }
        if (M.core_dock) {
            M.core_dock.init(Y);
        }
        new TREE(properties);
    }
};

}, '@VERSION@', {requires:['base', 'core_dock', 'io-base', 'node', 'dom', 'event-custom', 'event-delegate', 'json-parse']});
