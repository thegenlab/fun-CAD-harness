/* Pure workbench state helpers, shared by the UI and its tests. */
(() => {
  const status = value => ['not_started','preparing'].includes(value) ? 'preparing' : value;
  function filterProjects(projects, query, selectedStatus, summaries = {}) {
    const words = String(query || '').toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
    const matched = projects.filter(project => words.every(word => `${project.title || ''} ${project.id}`.toLocaleLowerCase().includes(word)));
    const counts = {all:matched.length,running:0,passed:0,failed:0,blocked:0,preparing:0};
    for (const project of matched) {
      const key=status(summaries[project.id]?.status || project.status || 'preparing');
      counts[key]=(counts[key] || 0)+1;
    }
    return {counts,items:matched.filter(project => selectedStatus==='all' || status(summaries[project.id]?.status || project.status || 'preparing')===selectedStatus)};
  }
  function searchEvents(items, query) {
    const words=String(query || '').toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
    return items.filter(item=>words.every(word=>`${item.title || ''} ${item.text || ''} ${item.output || ''}`.toLocaleLowerCase().includes(word)));
  }
  class Notices {
    constructor(seen={}) {this.seen=seen;this.previous=null;this.unread=new Map();}
    update(summary) {
      if (!summary.ready) return [];
      const current=new Map(summary.projects.map(project=>[project.id,project]));
      if (!this.previous) {this.previous=current;return [];}
      const fresh=[];
      for (const project of summary.projects) {
        const previous=this.previous.get(project.id);
        if (project.notify && previous?.notice_key!==project.notice_key && this.seen[project.id]!==project.notice_key) {
          this.seen[project.id]=project.notice_key;this.unread.set(project.id,project);fresh.push(project);
        }
      }
      this.previous=current;return fresh;
    }
    acknowledge(project) {this.unread.delete(project);}
  }
  function title(project, unread=0) {
    const icon={passed:'✓',failed:'✕',blocked:'!',running:'↻'}[project?.status] || '…';
    return `(${icon}${unread ? ` · ${unread}` : ''}) ${project?.title ? project.title+' · ' : ''}机械工作台`;
  }
  function age(at, now=Date.now()) {
    if (!at) return '时间未记录';
    const seconds=Math.max(0,Math.floor((now-at)/1000));
    return seconds<60 ? `${seconds} 秒前` : seconds<3600 ? `${Math.floor(seconds/60)} 分钟前` : `${Math.floor(seconds/3600)} 小时前`;
  }
  const api={filterProjects,searchEvents,Notices,title,age};
  if (typeof module!=='undefined') module.exports=api;
  if (typeof window!=='undefined') window.Workbench=api;
})();
