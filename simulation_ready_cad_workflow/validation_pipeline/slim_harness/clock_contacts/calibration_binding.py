"""Verify actual calibration evidence/runtime bindings before creating a recorder."""
from pathlib import Path
import hashlib
import json
from .runtime_identity import runtime_identity, sha256_file
from .contact_collector import RigidContactCollector


class CalibrationBindingError(RuntimeError):pass


def content_signature(value):
    payload={key:item for key,item in value.items() if key!='signature'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()


def bound_runtime_libraries(identity):
    # Real v2 probe metadata used this old field name for all selected loaded
    # prefix libraries, not just Chrono-owned libraries. Preserve that meaning.
    return identity.get('loaded_runtime_libraries',identity.get('loaded_chrono_native_components',[]))


def verify_saved_bundle(bundle_path,measurement_file):
    path=Path(bundle_path).resolve();bundle=json.loads(path.read_text(encoding='utf-8'))
    if bundle.get('schema_version')!='nsc_rigid_contact_calibration_v2' or bundle.get('signature')!=content_signature(bundle):
        raise CalibrationBindingError('CONTACT_CALIBRATION_BINDING_ERROR: calibration signature mismatch')
    if bundle.get('impact_event_rule')!='first_positive_impulse_classifies_episode':
        raise CalibrationBindingError('CONTACT_CALIBRATION_BINDING_ERROR: unreviewed impact episode rule')
    for binding in bundle['evidence_bindings']:
        artifact=(path.parent/binding['path']).resolve()
        if not artifact.is_relative_to(path.parent) or not artifact.is_file() or sha256_file(artifact)!=binding['sha256']:
            raise CalibrationBindingError('CONTACT_CALIBRATION_BINDING_ERROR: changed/missing evidence '+binding['path'])
    for name,expected in bundle['collector_implementations'].items():
        if sha256_file(Path(__file__).with_name(name))!=expected:
            raise CalibrationBindingError('CONTACT_CALIBRATION_BINDING_ERROR: collector implementation changed '+name)
    if sha256_file(measurement_file)!=bundle['measurement_implementation_sha256']:
        raise CalibrationBindingError('CONTACT_CALIBRATION_BINDING_ERROR: measurement implementation changed')
    return bundle


def verify_current_runtime(recorded,current):
    if recorded['core_wrapper']['sha256']!=current['core_wrapper']['sha256']:
        raise CalibrationBindingError('CONTACT_CALIBRATION_BINDING_ERROR: Chrono core wrapper changed')
    if recorded['active_chrono_package_records']!=current['active_chrono_package_records']:
        raise CalibrationBindingError('CONTACT_CALIBRATION_BINDING_ERROR: active Chrono package version/build changed')
    observed={row['path']:row['sha256'] for row in bound_runtime_libraries(current)}
    for required in bound_runtime_libraries(recorded):
        if observed.get(required['path'])!=required['sha256']:
            raise CalibrationBindingError('CONTACT_CALIBRATION_BINDING_ERROR: calibrated runtime library changed/missing '+required['path'])


def create_registered_collector(core,system,bodies,body_pairs,bundle_path,measurement_file=None):
    """Production entry; use returned schema in the trace metadata.

    Install these four small modules as slim_harness.clock_contacts. The bundle
    stays in the program-owned run-support area. Raw probe construction remains
    possible separately, but is not the D acceptance entry.
    """
    measurement_file=measurement_file or Path(__file__).resolve().parents[1]/'alarm_measurements.py'
    bundle=verify_saved_bundle(bundle_path,measurement_file)
    verify_current_runtime(bundle['runtime_identity'],runtime_identity(core))
    recorder=RigidContactCollector(core,system,bodies,body_pairs,bundle['units_review'])
    schema={**recorder.measurement_schema(),'calibration_signature':bundle['signature'],
            'collector_implementation_sha256':bundle['collector_implementations']['contact_collector.py']}
    return recorder,schema
