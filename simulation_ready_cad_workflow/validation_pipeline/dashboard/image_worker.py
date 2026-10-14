"""Render original GLB instances with one shared orthographic camera."""
import json
from pathlib import Path
import sys


VIEWS={'iso':(23,-58),'front':(0,-90),'side':(0,0),'top':(90,-90)}


def render(request):
    import numpy as np
    import trimesh
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    up=np.asarray(request.get('up_axis') or [0,1,0],dtype=float)
    transform=trimesh.geometry.align_vectors(up,[0,0,1])
    scenes=[];bounds=[]
    for filename in request['models']:
        scene=trimesh.load(filename,force='scene',process=False)
        meshes=[]
        for node in sorted(scene.graph.nodes_geometry,key=str):
            matrix,name=scene.graph[node];mesh=scene.geometry[name].copy()
            if not isinstance(mesh,trimesh.Trimesh) or not len(mesh.faces):continue
            mesh.apply_transform(matrix);mesh.apply_transform(transform)
            meshes.append(mesh);bounds.extend(mesh.bounds)
        if not meshes:raise ValueError('原始模型没有可绘制的表面')
        scenes.append(meshes)
    minimum=np.min(bounds,axis=0);maximum=np.max(bounds,axis=0)
    center=(minimum+maximum)/2;span=maximum-minimum;span=np.maximum(span,max(span.max(),.001)*.1)
    camera={'center_m':center.tolist(),'span_m':span.tolist(),'view':request.get('view','iso'),
            'projection':'orthographic','up_transform':transform.tolist()}
    output=Path(request['output']);output.mkdir(parents=True,exist_ok=True)
    paths=[]
    for index,meshes in enumerate(scenes):
        fig=plt.figure(figsize=(request['width']/100,request['height']/100),dpi=100,facecolor='#f6f7f3')
        ax=fig.add_subplot(111,projection='3d');ax.set_facecolor('#f6f7f3')
        triangles=[];colors=[];light=np.asarray([.4,-.3,.85]);light/=np.linalg.norm(light)
        for mesh in meshes:
            visual=mesh.visual.to_color() if hasattr(mesh.visual,'to_color') else mesh.visual
            rgba=np.asarray(visual.face_colors,dtype=float)/255
            if len(rgba)!=len(mesh.faces):rgba=np.tile([.5,.6,.65,1],(len(mesh.faces),1))
            rgba[:,:3]*=np.clip(.75+.25*(mesh.face_normals@light),.4,1)[:,None]
            triangles.append(mesh.triangles);colors.append(rgba)
        ax.add_collection3d(Poly3DCollection(np.concatenate(triangles),facecolors=np.concatenate(colors),edgecolors='none'))
        for setter,lo,hi in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center-span*.58,center+span*.58):setter(lo,hi)
        ax.set_box_aspect(span);ax.view_init(*VIEWS[camera['view']]);ax.set_proj_type('ortho');ax.set_axis_off()
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        target=output/f'{index}.png';temporary=output/f'{index}.pending.png'
        fig.savefig(temporary,dpi=100,facecolor=fig.get_facecolor());plt.close(fig);temporary.replace(target);paths.append(str(target))
    return {'status':'ready','images':paths,'camera':camera,'width':request['width'],'height':request['height']}


if __name__=='__main__':
    request=json.loads(Path(sys.argv[1]).read_text())
    try:result=render(request)
    except Exception as exc:result={'status':'failed','error':str(exc)}
    (Path(request['output'])/'result.json').write_text(json.dumps(result),encoding='utf-8')
    if result['status']!='ready':raise SystemExit(1)
