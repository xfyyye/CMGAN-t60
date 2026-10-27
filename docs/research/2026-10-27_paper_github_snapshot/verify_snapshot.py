from pathlib import Path
import collections,hashlib,json,re,subprocess,zipfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
RUN=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def dump(name,obj):(RUN/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

checks={}
zh=(ROOT/'docs/main_zh.md').read_text()
bi=(ROOT/'docs/main_bilingual.md').read_text()
tex=(ROOT/'docs/main.tex').read_text()
bib=(ROOT/'docs/refs.bib').read_text()
oldzh=(RUN/'before/main_zh.md').read_text()
oldbi=(RUN/'before/main_bilingual.md').read_text()
checks['exact_year_only_manuscript_edits']=(zh==oldzh.replace('Xiong 等（2018）','Xiong 等（2019）') and bi==oldbi.replace('Xiong 等（2018）','Xiong 等（2019）').replace('Xiong et al. (2018)','Xiong et al. (2019)'))
checks['old_year_absent_from_current_prose']=not any(x in s for s in [zh,bi] for x in ['Xiong 等（2018）','Xiong et al. (2018)'])
checks['correct_year_occurrence_counts']={'Chinese':zh.count('Xiong 等（2019）'),'bilingual_Chinese':bi.count('Xiong 等（2019）'),'bilingual_English':bi.count('Xiong et al. (2019)')}
assert list(checks['correct_year_occurrence_counts'].values())==[1,1,1]
xiong=re.search(r'@article\{xiong2018ercnn,(.*?)\n\}',bib,re.S).group(1)
checks['bib_year_and_historical_key_correct']=bool(re.search(r'year\s*=\s*\{2019\}',xiong)) and '\\citet{xiong2018ercnn}' in tex

image_sets=[]
for rel in ['docs/main_zh.md','docs/main_bilingual.md']:
    p=ROOT/rel
    refs=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',p.read_text())
    paths=[(p.parent/x).resolve() for x in refs]
    assert len(paths)==9 and all(x.is_file() for x in paths)
    image_sets.append(paths)
tex_images=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',tex)
tex_paths=[ROOT/'docs'/x for x in tex_images]
assert len(tex_paths)==9 and all(x.is_file() for x in tex_paths)
checks['images']={'Chinese':len(image_sets[0]),'bilingual':len(image_sets[1]),'LaTeX':len(tex_paths),'all_paths_exist':True,'all_three_ordered_sha256_lists_identical':len({tuple(sha(p) for p in ps) for ps in [*image_sets,tex_paths]})==1}
keys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib))
cited=set()
for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}',tex):cited.update(x.strip() for x in group.split(','))
checks['citations']={'unique_cited_keys':len(cited),'bib_keys':len(keys),'missing_keys':sorted(cited-keys)}
assert len(cited)==33 and not cited-keys
labels=re.findall(r'\\label\{([^}]+)\}',tex)
refs=re.findall(r'\\(?:eqref|ref|autoref)\*?\{([^}]+)\}',tex)
checks['cross_references']={'duplicate_labels':[x for x,n in collections.Counter(labels).items() if n>1],'undefined_refs':sorted(set(refs)-set(labels))}
assert not checks['cross_references']['duplicate_labels'] and not checks['cross_references']['undefined_refs']
counts={e:len(re.findall(r'\\begin\{'+e+r'\}',tex)) for e in ['equation','figure','table']}
counts['figure']+=len(re.findall(r'\\begin\{figure\*\}',tex))
checks['numbered_objects']=counts
assert counts=={'equation':25,'figure':8,'table':6},counts
stack=[]
for m in re.finditer(r'\\(begin|end)\{([^}]+)\}',tex):
    op,env=m.groups()
    if op=='begin':stack.append(env)
    else:
        assert stack and stack.pop()==env,('environment nesting',env)
assert not stack
checks['latex_environment_pairing']=True

with zipfile.ZipFile(ROOT/'docs/main_zh.docx') as z:
    doc=ET.fromstring(z.read('word/document.xml'))
    wordtext=''.join(doc.itertext())
    checks['Word_has_Xiong_2019']='Xiong 等（2019）' in wordtext
    media=[hashlib.sha256(z.read(x)).hexdigest() for x in z.namelist() if x.startswith('word/media/')]
    checks['Word_images_match_current_sources']=collections.Counter(media)==collections.Counter(sha(x) for x in image_sets[0])

protected=json.loads((RUN/'protected_hashes_before.json').read_text())
checks['protected_file_hashes_unchanged']={p:sha(ROOT/p)==h for p,h in protected.items() if p!='docs/HANDOVER.md'}
checks['original_RIR_handover_preserved']=(ROOT/'docs/HANDOVER.md').read_text().endswith((RUN/'before/HANDOVER.md').read_text().split('\n\n',1)[1])
assert all(checks['protected_file_hashes_unchanged'].values())
assert checks['original_RIR_handover_preserved']
assert checks['exact_year_only_manuscript_edits'] and checks['bib_year_and_historical_key_correct']
assert checks['images']['all_three_ordered_sha256_lists_identical'] and checks['Word_has_Xiong_2019'] and checks['Word_images_match_current_sources']

plan=json.loads((RUN/'stage_plan.json').read_text())
scientific=plan[:29]
scientific+=['docs/figs/dataset_distribution_train.png','docs/figs/dataset_distribution_test.png']
files={p:{'sha256':sha(ROOT/p),'bytes':(ROOT/p).stat().st_size} for p in scientific}
assert max(x['bytes'] for x in files.values())<50*1024*1024
base=(RUN/'head_before.txt').read_text().strip()
manifest={'base_commit':base,'branch':'exp/kan-multitask','scientific_files':files,'full_staged_paths':plan,'partial_staged_path':'docs/HANDOVER.md','partial_staged_content':'paper status section only; local RIR sections excluded','image_correspondence':[{'Chinese_source':str(a.relative_to(ROOT)),'bilingual_source':str(b.relative_to(ROOT)),'LaTeX_asset':str(c.relative_to(ROOT)),'sha256':sha(a)} for a,b,c in zip(*image_sets,tex_paths)]}
dump('manifest.json',manifest)
patch=git('diff',base,'--','docs/main_zh.md','docs/main_bilingual.md','docs/main.tex','docs/refs.bib').decode()+(RUN/'handover_paper_only.patch').read_text()
(RUN/'snapshot_from_previous_commit.diff').write_text(patch)
r=subprocess.run(['git','diff','--check','--','docs/main_zh.md','docs/main_bilingual.md','docs/main.tex','docs/refs.bib'],cwd=ROOT,capture_output=True,text=True)
checks['git_diff_whitespace']={'exit_code':r.returncode,'output':r.stdout+r.stderr}
assert r.returncode==0
checks['latex_compiled']=False
checks['Word_rerendered_this_run']=False
checks['scope']='Year edits, scientific asset/citation/label dependencies, LaTeX environment pairing and object counts, protected worktree bytes, and whitespace only; no full translation or runtime validation.'
checks['status']='PASS'
dump('checks.json',checks)
print(json.dumps({'status':'PASS','files_in_manifest':len(files),'images':9,'cited_keys':len(cited),'numbered_objects':counts,'protected_files_unchanged':len(checks['protected_file_hashes_unchanged'])},ensure_ascii=False))
