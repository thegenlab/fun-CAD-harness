"""公共执行器的原生物理单元模型；不是46类CAD的替代样本。"""
from pathlib import Path
import json
import argparse
import math
from simulation_ready_cad_workflow.validation_pipeline.mechanism.pychrono_core import load_pychrono_core
ch = load_pychrono_core()
from .native import NativeModel, Signal, Actuator, Operation, run


def native_body(kind):
    system=ch.ChSystemNSC()
    system.SetGravitationalAcceleration(ch.ChVector3d(0,0,0))
    frame=ch.ChBody();frame.SetName('frame');frame.SetFixed(True);system.Add(frame)
    body=ch.ChBody();body.SetName('moving');body.SetMass(1.0)
    body.SetInertiaXX(ch.ChVector3d(1,1,1));system.Add(body)
    joint=(ch.ChLinkLockPrismatic() if kind=='body_force' else ch.ChLinkLockRevolute())
    joint.Initialize(frame,body,ch.ChFramed());system.Add(joint)
    signal_kind='body_position' if kind=='body_force' else 'body_angle'
    return NativeModel(system,{'output':Signal(body,signal_kind,reference=frame)},
                       {'handle':Actuator(body,kind,reference=frame)},keepalive=(frame,body,joint),
                       scope='native API unit model only; not benchmark CAD')


def shaft_model():
    system=ch.ChSystemNSC();shaft=ch.ChShaft();shaft.SetInertia(1);system.Add(shaft)
    return NativeModel(system,{'output':Signal(shaft,'shaft_angle')},
                       {'handle':Actuator(shaft,'shaft')},keepalive=shaft,
                       scope='native API unit model only; not benchmark CAD')


def offcenter_force_model():
    # 刚体质心被转轴固定，但真实受力点会移动；检查不能误用质心坐标/速度。
    system=ch.ChSystemNSC();system.SetGravitationalAcceleration(ch.ChVector3d(0,0,0))
    frame=ch.ChBody();frame.SetFixed(True);system.Add(frame)
    body=ch.ChBody();body.SetMass(1);body.SetInertiaXX(ch.ChVector3d(1,1,1));system.Add(body)
    link=ch.ChLinkLockRevolute()
    link.Initialize(frame,body,ch.ChFramed(ch.ChVector3d(0,0,0),ch.QuatFromAngleAxis(-math.pi/2,ch.ChVector3d(1,0,0))))
    system.Add(link)
    port=Actuator(body,'body_force',reference=frame,point_local=(1,0,0),axis=(0,0,1))
    model=NativeModel(system,{'output':Signal(port.coordinate,'source_scalar')},{'handle':port},
                       keepalive=(frame,body,link),scope='off-center force native unit probe, not benchmark CAD')
    return model


def hanging_load_model():
    from .fixtures import add_hanging_cable_load
    system=ch.ChSystemNSC();system.SetGravitationalAcceleration(ch.ChVector3d(0,0,-9.81))
    frame=ch.ChBody();frame.SetFixed(True);system.Add(frame)
    drum=ch.ChBody();drum.SetMass(1.);drum.SetInertiaXX(ch.ChVector3d(.1,.1,.1));system.Add(drum)
    joint=ch.ChLinkLockRevolute()
    joint.Initialize(drum,frame,ch.ChFramed(ch.ChVector3d(0,0,0),ch.QuatFromAngleAxis(-math.pi/2,ch.ChVector3d(1,0,0))))
    system.Add(joint)
    model=NativeModel(system,{}, {'handle':Actuator(drum,'body_torque',reference=frame,axis=(0,1,0))},
        keepalive=(frame,drum,joint),scope='hanging load API unit model, not benchmark CAD')
    model.objects={'bodies':{'frame':frame,'drum':drum}}
    fixture=add_hanging_cable_load(model,drum_id='drum',frame_id='frame',axis=(0,1,0),
        radius_m=.03,mass_kg=1.,initial_center=(-.03,0,-.5))
    model.signals={'output':fixture.drum_angle,'load_height':Signal(fixture.body,'body_position',axis=(0,0,1)),
                   'load_speed':Signal(fixture.body,'body_speed',axis=(0,0,1)),
                   'drum_speed':fixture.drum_speed,'tension':Signal(lambda:fixture.tension_n,'source_scalar')}
    return model


def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    records=[]
    for kind in ('shaft','body_force','body_torque','offcenter_force'):
        model=(shaft_model() if kind=='shaft' else offcenter_force_model() if kind=='offcenter_force' else native_body(kind))
        result=run(model,[Operation('handle',0,.1,'effort',1.,1.)],duration_s=.2,
                   step_s=.001,output=output/kind,required_signals=('output',),sample_times=(.1,.15))
        assert result['status']=='completed'
        assert result['final']['output']>result['release_state']['output']>0
        # 每步及时提交人工力，释放后不应多施加一小步的旧载荷。
        assert abs(result['delta_after_release']['output']-.01)<2e-6
        records.append({'kind':kind,'time_s':result['actual_time_s'],'steps':result['steps'],
                        'after_release_motion':result['delta_after_release']['output']})
    model=hanging_load_model()
    result=run(model,[Operation('handle',0,.1,'effort',.8,.8)],duration_s=.2,
               step_s=.001,output=output/'hanging_load',required_signals=('output','load_height'))
    assert result['final']['output']>0
    assert result['final']['load_height']>result['initial']['load_height']
    assert result['final']['tension']>0
    records.append({'kind':'hanging_load','time_s':result['actual_time_s'],'steps':result['steps'],
                    'actual_lift_m':result['final']['load_height']-result['initial']['load_height']})
    model=shaft_model()
    result=run(model,[Operation('handle',0,.1,'speed',0.,1.,gain=0.,bias_effort=1.)],duration_s=.2,
               step_s=.001,output=output/'feedforward_release',required_signals=('output',))
    assert abs(result['delta_after_release']['output']-.01)<2e-6
    records.append({'kind':'feedforward_release','time_s':result['actual_time_s'],'steps':result['steps'],
                    'after_release_motion':result['delta_after_release']['output']})
    (output/'checks.json').write_text(json.dumps({'is_benchmark_case':False,'native_probes':records},indent=2))
    print(json.dumps(records))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
