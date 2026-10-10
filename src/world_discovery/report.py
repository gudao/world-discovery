from pathlib import Path
import html
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_report(out, summaries, experiment):
    out = Path(out)
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = sorted(set((r['policy'], r['memory']) for r in summaries))
    for policy, mode in groups:
        rows = sorted([r for r in summaries if r['policy'] == policy and r['memory'] == mode], key=lambda r: r['budget'])
        ax.errorbar([r['budget'] for r in rows], [r['mean_nmse'] for r in rows],
                    yerr=[r['world_standard_error'] for r in rows], label=f'{policy}/{mode}', marker='o', capsize=2)
    ax.set(xlabel='Number of real probes', ylabel='Normalized prediction MSE', title=f'Experiment {experiment}: paired-world evaluation')
    ax.set_yscale('symlog', linthresh=0.01)
    ax.legend(ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / 'learning_curve.png', dpi=160)
    plt.close(fig)
    body = ''.join(f"<tr><td>{html.escape(r['policy'])}/{html.escape(r['memory'])}</td><td>{r['budget']}</td><td>{r['mean_nmse']:.6f}</td><td>{r['world_standard_error']:.6f}</td></tr>" for r in summaries)
    summary = json.loads((out / 'summary.json').read_text())
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>World Discovery — {experiment}</title>
    <style>body{{max-width:1000px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#23324c}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}code{{background:#eef3fa;padding:3px}}</style>
    <h1>世界发现实验 {experiment}</h1><p>MuJoCo 数值感知 · 主动探索 · 快速局部记忆</p>
    <p>最终预算下，随机减主动的配对 NMSE 差：<b>{summary['paired_random_minus_active']:.6f}</b>。正值表示本次主动策略更好，负值表示更差。</p>
    <p>误差条为一个随机种子内世界间的标准误，不能替代多种子统计。容量扩展不是发现物理概念的证明。</p>
    <img src="learning_curve.png" alt="学习曲线"><h2>结果</h2>
    <table><tr><th>策略 / 记忆</th><th>真实试探次数</th><th>平均 NMSE</th><th>世界间标准误</th></tr>{body}</table>
    <h2>审计文件</h2><p><code>metrics.csv</code>：逐世界指标；<code>interactions.jsonl</code>：真实交互；<code>growth_events.json</code>：扩展决策；<code>manifest.json</code>：版本和哈希。</p></html>'''
    (out / 'report.html').write_text(content, encoding='utf-8')
