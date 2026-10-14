"""Memory available to this host/container, without a machine-specific RSS cap."""
from __future__ import annotations
import os
from pathlib import Path
import psutil


def cgroup_memory_files(cgroup_file=Path("/proc/self/cgroup"), mount_file=Path("/proc/self/mountinfo")):
    """Read this process's v1/v2 hierarchy, including tighter ancestor limits."""
    if not cgroup_file.is_file() or not mount_file.is_file():return []
    memberships={}
    for line in cgroup_file.read_text().splitlines():
        _,controllers,path=line.split(":",2)
        if not controllers:memberships["cgroup2"]=Path(path)
        elif "memory" in controllers.split(","):memberships["cgroup"]=Path(path)
    pairs=[]
    def unescape(value):
        for encoded,decoded in ((r"\040"," "),(r"\011","\t"),(r"\134","\\")):value=value.replace(encoded,decoded)
        return value
    for line in mount_file.read_text().splitlines():
        before,after=line.split(" - ",1);left=before.split();right=after.split()
        kind=right[0]
        if kind not in memberships:continue
        if kind=="cgroup" and "memory" not in right[2].split(","):continue
        mount=Path(unescape(left[4]));root=Path(unescape(left[3]));member=memberships[kind]
        try:current=mount/member.relative_to(root)
        except ValueError:current=mount/member.relative_to(member.anchor)  # Namespace-relative membership.
        names=("memory.max","memory.current") if kind=="cgroup2" else ("memory.limit_in_bytes","memory.usage_in_bytes")
        while current==mount or mount in current.parents:
            pair=(current/names[0],current/names[1])
            if pair not in pairs:pairs.append(pair)
            if current==mount:break
            current=current.parent
    return pairs


def available_memory(*,memory=None,cgroup_files=None):
    memory=memory if memory is not None else psutil.virtual_memory()
    total=int(memory.total);available=min(total,int(memory.available));constraints=[]
    pairs=cgroup_files if cgroup_files is not None else cgroup_memory_files() if os.name=="posix" else []
    for limit_file,used_file in pairs:
        if not limit_file.is_file():continue
        raw=limit_file.read_text().strip()
        if raw=="max":continue
        limit=int(raw)
        if limit>=2**60:continue  # v1 unlimited sentinel
        used=int(used_file.read_text().strip())
        total=min(total,limit);available=min(available,max(0,limit-used))
        constraints.append({"limit_bytes":limit,"used_bytes":used,"path":str(limit_file)})
    return {"host_total_bytes":int(memory.total),"host_available_bytes":int(memory.available),
            "effective_total_bytes":total,"effective_available_bytes":max(0,available),"cgroup_constraints":constraints}


class AutoMemoryLimit:
    """Reserve 20% of initially available RAM; stop only our child under pressure."""
    def __init__(self):
        snapshot=available_memory();available=snapshot["effective_available_bytes"]
        if available<=0:raise ValueError("No available memory in the current host/container")
        self.reserve=max(1,int(available*.2));self.initial=max(1,available-self.reserve)
        self.report={"mode":"auto","initial":snapshot,"reserve_bytes":self.reserve,
                     "initial_max_rss_bytes":self.initial,"effective_max_rss_bytes":self.initial}

    def limit_for_rss(self,rss):
        current=available_memory()
        # Available RAM excludes this running tree's pages. Add its current RSS
        # back before comparing, so its own allocations do not halve its cap.
        limit=max(0,min(self.initial,int(rss)+current["effective_available_bytes"]-self.reserve))
        self.report.update(last=current,effective_max_rss_bytes=limit)
        return limit
