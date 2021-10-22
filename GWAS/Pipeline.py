import numpy as np
import h5py
from contextlib import contextmanager, ExitStack
from multiprocessing import Process, Pool, Manager
from os import system
from itertools import product
import pandas as pd
from collections import defaultdict
from pandas.errors import EmptyDataError
from functools import reduce
import re
from random import sample
import string
from glob import glob
from matplotlib import pyplot as plt
import requests
import pickle as pkl


ukb_bed_path = ""
ukb_pheno_path = ""
ukb_imputed_path = ""
plink_path = ""
bolt_path = ""
LD_score_path = ""
hg19_path = ""
p = "P_BOLT_LMM_INF"
LDlink_token = "114f40477521"
WINDOW_SIZE = 200000


# cohort class, stores phenotypes, writes phenotypes and covariate variables
class Cohort:
    def __init__(self, cohort_name, eids=[], features=[], covar_path=None, pheno_path=None, hap_chr_path=None, hap_chrall_path=None,
                 merge_list_path=None, stat_path=None, log_path=None, base_path='./'):
        assert len(features) == len(eids), "length of features and length of ids do not match"
        if len(eids) > 0:
            assert all(list(map(lambda x: isinstance(x, str), eids))), 'eids must be string'
        self.additional_covar_list = []
        self.name = cohort_name
        self.len = len(eids)
        self.num_phenos = None if (len(features))==0 else (1 if isinstance(features[0], int) else len(features[0]))
        self.eids = dict(zip(eids, range(self.len)))
        self.features = list(features)
        if base_path[-1] != '/':
            base_path += '/'
        self.covar_path = f"{base_path}{self.name}_covar" if not covar_path else covar_path
        self.pheno_path = f"{base_path}{self.name}_pheno" if not pheno_path else pheno_path
        self.hap_chr_path = f"{base_path}ukb_hap_chr{{}}_{self.name}" if not hap_chr_path else hap_chr_path
        self.hap_chrall_path = f"{base_path}ukb_hap_chrall_{self.name}" if not hap_chrall_path else hap_chrall_path
        self.merge_list_path = f"{base_path}{self.name}_merge_list" if not merge_list_path else merge_list_path
        self.stat_path = f"{base_path}stat_{self.name}_{{}}" if not stat_path else stat_path
        self.log_path = f"{base_path}out_{self.name}_{{}}.log" if not log_path else log_path

    @contextmanager
    def open_pheno(self):
        self.fpheno = open(self.pheno_path, 'w')
        self.fpheno.write("FID IID ")
        buffer = ""
        for i in range(self.num_phenos):
            buffer += f"QT{i} "
        self.fpheno.write(buffer[:-1])
        self.fpheno.write("\n")
        try:
            yield
        finally:
            self.fpheno.close()
    
    def overwrite_covar_header(self, additional_covar_list=[]):
        self.additional_covar_list = additional_covar_list
        if not self.fcovar.closed:
            self.fcovar.close()
        self.fcovar = open(self.covar_path, 'w')        
        self.fcovar.write("FID IID SEX AGE GE")
        for i in range(10):
            self.fcovar.write(f" PC{i}")
        self.fcovar.write(" ETH")
        for covar in additional_covar_list:
            self.fcovar.write(" "+covar)
        self.fcovar.write('\n')        

    @contextmanager
    def open_files(self, covar_path=None, pheno_path=None):
        if covar_path:
            self.covar_path = covar_path
        if pheno_path:
            self.pheno_path = pheno_path
        self.fcovar = open(self.covar_path, 'w')
        self.fpheno = open(self.pheno_path, 'w')
        self.fpheno.write("FID IID ")
        buffer = ""
        for i in range(self.num_phenos):
            buffer += f"QT{i} "
        self.fpheno.write(buffer[:-1])
        self.fpheno.write("\n")
        self.fcovar.write("FID IID SEX AGE GE ")
        for i in range(10):
            self.fcovar.write(f"PC{i} ")
        self.fcovar.write("ETH")
        self.fcovar.write('\n')
        try:
            yield
        finally:
            self.fcovar.close()
            self.fpheno.close()

    def __contains__(self, key):
        return key in self.eids
    
    def __getitem__(self, key):
        return self.features[self.eids[key]]
    
    def __setitem__(self, key, val):
        assert isinstance(key, str), 'eid must be string'
        if self.num_phenos:
            assert len(val) == self.num_phenos, 'wrong number of phenotypes'
        else:
            self.num_phenos = len(val)
        try:
            self.features[self.eids[key]] = val
        except:
            self.eids[key] = self.len
            self.len += 1
            self.features.append(val)
    
    def write_pheno(self, eid):
        self.fpheno.write(f"{eid} {eid} ")
        phenos = self.features[self.eids[eid]]
        buffer = ""
        for pheno in phenos:
            buffer += f"{pheno} "
        self.fpheno.write(buffer[:-1])
        self.fpheno.write('\n')
        
        
    def manhattan_plot(figsize=(20, 15), markersize=5, fontsize=20, annotation_snp_dict=None, save_path=None):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        neglogp_max = 0
        for i, stat_file in enumerate([self.stat_path.format(x) for x in range(self.num_phenos)]):
            try:
                data = pd.read_csv(stat_file, '\t')
            except (FileNotFoundError, EmptyDataError):
                continue
            data['-log10(p_value)'] = -np.log10(data['P_BOLT_LMM_INF'])
            neglogp_max = data['-log10(p_value)'].max() if data['-log10(p_value)'].max() > neglogp_max else neglogp_max
            data['CHR'] = data['CHR'].astype('category')
            if i == 0:
                chlen = np.concatenate([np.array([0]), data.groupby("CHR").BP.max().cumsum().values[:-1]])
            data['ind'] = data["BP"].values + chlen[data["CHR"].values.astype('i') - 1]
            colors = ['#E24E42', '#008F95']
            data_grouped = data.groupby(('CHR'))
            significance = -np.log10(5e-8)
            x_labels = []
            x_labels_pos = []
            margin = 1
            for num, (name, group) in enumerate(data_grouped):
                group.plot(kind='scatter', x='ind', y='-log10(p_value)', logy=True, color=colors[num % len(colors)], ax=ax, s=150)
                if i == 0:
                    x_labels.append(name)
                    x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0]) / 2))
            if i == 0:
                ax.set_xticks(x_labels_pos)
                ax.set_xticklabels(x_labels)
                ax.set_xlim([0, data.ind.max()])
                ax.set_xlabel('Chromosome', fontsize=40)
                ax.set_ylabel('$-\log_{10}(P)$', fontsize=40)
                plt.axhline(y=significance, color='gray', linestyle='-', linewidth = 2)
                plt.xticks(fontsize=fontsize, rotation=60)
                plt.yticks(fontsize=fontsize)
        ax.set_ylim([1, min(neglogp_max, 325) + 1])

        if annotation_snp_dict is not None:
            for index, row in data.iterrows():
                if row['SNP'] in annotation_snp_dict and row["-log10(p_value)"] > significance:
                    ax.annotate(annotation_snp_dict[row["SNP"]], xy = (index, row["-log10(p_value)"]), fontsize=20)
        if save_path:
            plt.savefig(save_path, quality=100)
        plt.show()

    def write_covar(self, eid, sex, age, ge, pcs, eth, additional_covar_list=[]):
        self.fcovar.write(f"{eid} {eid} {sex} {age} {ge}")
        for pc in pcs:
            self.fcovar.write(f" {pc}")
        self.fcovar.write(f" {eth}")
        for covar in additional_covar_list:
            self.fcovar.write(f" {covar}")
        self.fcovar.write('\n')
    
    def make_bed(self):
        for chrom in range(1, 23):
            system(f"{plink_path} --bfile {ukb_bed_path.format(chrom)} --make-bed\
                   --keep {self.pheno_path} --out {self.hap_chr_path.format(chrom)}")

    def merge_bed(self):
        with open(f"{self.merge_list_path}", 'w') as f:
            for i in range(1, 23):
                f.write(f"{self.hap_chr_path.format(i)}\n")
        system(f"{plink_path} --merge-list {self.merge_list_path} --out {self.hap_chrall_path}")
        system(f"rm {self.hap_chr_path.format('[0-9]*')+'*'}")
    
    def gwas(self, criterion=lambda **kwargs: True, additional_covar_list=[], additional_covar_dict={}):
        if not glob(self.hap_chrall_path.format('*')+'.bed'):
               self.make_bed()
               self.merge_bed()
        if not glob(self.covar_path.format('*')):
            filter_and_write([self], [criterion], additional_covar_list, additional_covar_dict)
        qcovar = ' '.join(f"--qCovarCol {x}" for x in self.additional_covar_list)
        for dim in range(self.num_phenos):
            system(f"{bolt_path} --numThreads 24 --bfile {self.hap_chrall_path} --phenoFile {self.pheno_path} --phenoCol "+\
                   f"QT{dim} --covarFile {self.covar_path} --covarCol SEX --covarCol GE --covarCol ETH --covarMaxLevels 22 --qCovarCol AGE"+\
                   f" --qCovarCol PC{{0:9}} {qcovar} --lmm --statsFile {self.stat_path.format(dim)} --LDscoresFile "+\
                   f"{LD_score_path} --lmmInfOnly --LDscoresMatchBp 2>&1|tee {self.log_path.format(dim)}")
    
    
    def extract_snp_pheno_pairs(self, thres):
        s = []
        for dim in range(self.num_phenos):
            try:
                df = pd.read_csv(self.stat_path.format(dim), '\t')[["SNP", "CHR", "BP", "P_BOLT_LMM_INF"]]
            except (FileNotFoundError, EmptyDataError):
                continue
            subset = df[df[p] < thres].copy()
            if len(subset) > 0:
                subset["DIM"] = dim
                s.append(subset)
        return pd.concat(s)
               

# get the sample info of UKBiobank, get covariate variables and do additional filtering
def filter_and_write(cohorts, criteria, additional_covar_list=[], additional_covar_dict={}):
    def na_check(cov):
        return cov if cov else "NA"
    with ExitStack() as stack:
        f = stack.enter_context(open(ukb_pheno_path, 'rb'))
        for cohort in cohorts:
            stack.enter_context(cohort.open_files())
        if additional_covar_list:
            for cohort in cohorts:
               cohort.overwrite_covar_header(additional_covar_list)
        header = f.readline().decode("utf-8", "ignore").strip().split(',')
        header_dict = dict(zip(map(lambda x: x.strip('"'), header), range(len(header))))
        for line in f:
            line = line.decode("utf-8", "ignore").strip().split(',')
            eid = line[0].strip('"')
            covar = {}
            if any([eid in cohort for cohort in cohorts]):
                covar["eth"] = na_check(line[header_dict["21000-0.0"]].strip('"'))
                covar["sex"] = na_check(line[header_dict["31-0.0"]].strip('"'))
                covar["age"] = na_check(line[header_dict["34-0.0"]].strip('"'))
                covar["ge"] = na_check(line[header_dict["22006-0.0"]].strip('"'))
                if "NA" in covar.values():
                    continue
                covar["pcs"] = [None]*10
                for i in range(10):
                    covar["pcs"][i] = na_check(line[header_dict["22009-0.{}".format(i+1)]].strip('"'))
                if "NA" in covar["pcs"]:
                    continue
                if eid in additional_covar_dict:
                    # print(eid)
                    covar["additional_covar_list"] = additional_covar_dict[eid]
                for cohort, criterion in zip(cohorts, criteria):
                    if eid in cohort and criterion(**covar):
                        cohort.write_covar(eid, **covar)
                        cohort.write_pheno(eid)


# some useful util functions
def merge_df(d, how="inner"):
    def join(x, y):
        x = x.set_index(["SNP", "CHR", "BP", "DIM"])
        y = y.set_index(["SNP", "CHR", "BP", "DIM"])
        d = x.merge(y, how=how, left_index=True, right_index=True)
        px = p + "_x"
        py = p + "_y"
        d = d.fillna(1)
        idx = (d[px]>d[py]).astype('i')
        d[p] = d[[px, py]].values[range(len(d)), idx]
        return d[p].reset_index()
    return reduce(lambda x, y: join(x, y), d)

def nms(snp_pheno_pairs):  #non-maximal suppression
    reduced = snp_pheno_pairs[snp_pheno_pairs[p] == snp_pheno_pairs.groupby("SNP")[p].transform(min)]
    reduced = reduced.drop_duplicates(subset=["SNP"])
    reduced.index = range(len(reduced))
    return reduced

def snp_to_gene(snps, snps_csv_path=None, gene_report_path=None, gene_list_border=2):
    del_flag1 = False
    del_flag2 = False
    alphabet = string.ascii_letters
    if not snps_csv_path:
        candidate_path = ''.join(sample(alphabet, 6))
        while glob(candidate_path):
            candidate_path = ''.join(sample(alphabet, 6))
        snps_csv_path = candidate_path
        del_flag1 = True
    if not gene_report_path:
        candidate_path = ''.join(sample(alphabet, 6))
        while glob(candidate_path):
            candidate_path = ''.join(sample(alphabet, 6))
        gene_report_path = candidate_path
        del_flag2 = True
    snps.to_csv(snps_csv_path, '\t', index=False)
    system(f"{plink_path} --gene-report {snps_csv_path} {hg19_path} --gene-list-border {gene_list_border} --out {gene_report_path}")
    snp_to_gene = defaultdict(set)
    with open(f"{gene_report_path}.range.report", 'r') as f:
        for line in f:
            gene = re.match("^((\w|-)+) ", line)
            if gene:
                curr_gene = gene.group(1)
            else:
                line = line.strip(' \n')
                if line and not re.match("^ *DIST", line):
                    dist, snp = line.split('\t')[0].split(' ')
                    snp_to_gene[snp].add((curr_gene, dist))
    if del_flag1:
        system(f"rm {snps_csv_path}")
    if del_flag2:
        system(f"rm {gene_report_path}.log")
        system(f"rm {gene_report_path}.range.report")
    return snp_to_gene

"""
def finemapping(snps, cohort, imputed_snp_path, window_size=WINDOW_SIZE, num_worker=24):
    range_path = f"{imputed_snp_path}{{}}_range"
    snp_path = f"{imputed_snp_path}{{}}"
    merge_list_path = f"{imputed_snp_path}{{}}_merge_list"
    merged_snp_path = f"{imputed_snp_path}{{}}_merged"
    snp_stat_path = f"{imputed_snp_path}stat_{{}}_{{}}"
    snp_log_path = f"{imputed_snp_path}{{}}.log"
    def f(queue):
        while True:
            element = queue.get()
            if element is None:
                return
            snp, bp, nchr = element
            with open(f"{range_path.format(snp)}", 'w') as f:
                f.write(f"{nchr} {max(1, bp-window_size)} {bp+window_size} 0")
            system(f"{plink_path} --bfile {ukb_imputed_path.format(nchr)} --extract range {range_path.format(snp)}\
                   --keep {cohort.covar_path} --make-bed --out {snp_path.format(snp)}")
            with open(f"{merge_list_path.format(snp)}", 'w') as f:
                for i in range(1, 23):
                    if i == int(nchr):
                        continue
                    f.write(f"{ukb_bed_path.format(i)}\n")
                f.write(f"{snp_path.format(snp)}\n")
            system(f"{plink_path} --merge-list {merge_list_path.format(snp)} --keep {cohort.covar_path}\
                   --make-bed --out {merged_snp_path.format(snp)}")
            if glob(f"{merged_snp_path.format(snp)}-merge.missnp"):
                system(f"{plink_path} --bfile {ukb_imputed_path.format(nchr)} --extract range {range_path.format(snp)}\
                       --exclude {merged_snp_path.format(snp)}-merge.missnp --keep {cohort.covar_path} --make-bed --out {snp_path.format(snp)}")
                system(f"{plink_path} --merge-list {merge_list_path.format(snp)} --keep {cohort.covar_path}\
                       --make-bed --out {merged_snp_path.format(snp)}")
    mng = Manager()
    queue = mng.Queue()
    pool = []
    snp_dim_dict = {}
    for _ in range(num_worker):
        process = Process(target=f, args=(queue,))
        process.start()
        pool.append(process)
    for _, row in snps.iterrows():
        snp = row["SNP"]
        nchr = row["CHR"]
        bp = row["BP"]
        dim = row["DIM"]
        snp_dim_dict[snp] = dim
        queue.put((snp, bp, nchr))
        for _ in range(num_worker):
            queue.put(None)
    for process in pool:
        process.join()
    for snp, dim in snp_dim_dict.items():
        system(f"{bolt_path} --numThreads 24 --bfile {merged_snp_path.format(snp)} --phenoFile {cohort.pheno_path} --phenoCol "+
               f"QT{dim} --covarFile {cohort.covar_path} --covarCol SEX --covarCol GE --covarCol ETH --covarMaxLevels 22 --qCovarCol AGE"+
               f" --qCovarCol PC{{0:9}} --lmmInfOnly --statsFile {snp_stat_path.format(snp, dim)} --LDscoresFile "+
               f"{LD_score_path} --LDscoresMatchBp 2>&1|tee {snp_log_path.format(snp)}")
        result = pd.read_csv(snp_stat_path.format(snp, dim), sep='\t')
        snps_in_range = pd.read_csv(snp_path.format(snp)+'.bim', sep='\t', header=None)[1]
        result[result["SNP"].isin(snps)].to_csv(snp_stat_path.format(snp, dim), sep='\t', index=False)
"""
def finemapping(snps, cohort, imputed_snp_path='./', window_size=WINDOW_SIZE):
    range_path = f"{imputed_snp_path}{{}}_range"
    snp_path = f"{imputed_snp_path}{{}}"
    merge_list_path = f"{imputed_snp_path}{{}}_merge_list"
    merged_snp_path = f"{imputed_snp_path}{{}}_merged"
    snp_stat_path = f"{imputed_snp_path}stat_{{}}_{{}}"
    snp_log_path = f"{imputed_snp_path}{{}}.log"
    snp_dim_dict = {}
    for _, row in snps.iterrows():
        snp = row["SNP"]
        nchr = row["CHR"]
        bp = row["BP"]
        dim = row["DIM"]
        snp_dim_dict[snp] = dim
        if glob(merged_snp_path.format(snp)+'.bed'):
            continue
        with open(f"{range_path.format(snp)}", 'w') as f:
            f.write(f"{nchr} {max(1, bp-window_size)} {bp+window_size} 0")
        system(f"{plink_path} --bfile {ukb_imputed_path.format(nchr)} --extract range {range_path.format(snp)}\
               --keep {cohort.covar_path} --make-bed --out {snp_path.format(snp)}")
        system(f"mv {snp_path.format(snp)}.log plink_{snp_path.format(snp)}.log")
        with open(f"{merge_list_path.format(snp)}", 'w') as f:
            for i in range(1, 23):
                if i == int(nchr):
                    continue
                f.write(f"{ukb_bed_path.format(i)}\n")
            f.write(f"{snp_path.format(snp)}\n")
        system(f"{plink_path} --merge-list {merge_list_path.format(snp)} --keep {cohort.covar_path}\
               --make-bed --out {merged_snp_path.format(snp)}")
        if glob(f"{merged_snp_path.format(snp)}-merge.missnp"):
            system(f"{plink_path} --bfile {ukb_imputed_path.format(nchr)} --extract range {range_path.format(snp)}\
                   --exclude {merged_snp_path.format(snp)}-merge.missnp --keep {cohort.covar_path} --make-bed --out {snp_path.format(snp)}")
            system(f"{plink_path} --merge-list {merge_list_path.format(snp)} --keep {cohort.covar_path}\
                   --make-bed --out {merged_snp_path.format(snp)}")
    for snp, dim in snp_dim_dict.items():
        system(f"{bolt_path} --numThreads 24 --bfile {merged_snp_path.format(snp)} --phenoFile {cohort.pheno_path} --phenoCol "+
               f"QT{dim} --covarFile {cohort.covar_path} --covarCol SEX --covarCol GE --covarCol ETH --covarMaxLevels 22 --qCovarCol AGE"+
               f" --qCovarCol PC{{0:9}} --lmmInfOnly --statsFile {snp_stat_path.format(snp, dim)} --LDscoresFile "+
               f"{LD_score_path} --LDscoresMatchBp 2>&1|tee {snp_log_path.format(snp)}")
        result = pd.read_csv(snp_stat_path.format(snp, dim), sep='\t')
        snps_in_range = pd.read_csv(snp_path.format(snp)+'.bim', sep='\t', header=None)[1]
        result[result["SNP"].isin(snps_in_range)].to_csv(snp_stat_path.format(snp, dim)+'_processed', sep='\t', index=False)
        


class RBI:
    """
    Red black tree that stores intervals,
    Only implemented the put and contains method, no delete
    key is the left end of the interval
    val is the right end of the interval
    """
    RED = 1
    BLACK = 0
    class Node:
        def __init__(self, key, val, N, color, **kwargs):
            assert key <= val, "not a valid interval"
            self.key = key
            self.val = val
            self.N = N
            self.color = color
            self.left = None
            self.right = None
            self.__dict__.update(kwargs)

    @staticmethod
    def isRed(x):
        if not x:
            return False
        return x.color == RBI.RED

    @staticmethod
    def size(x):
        if not x:
            return 0
        return x.N

    @staticmethod
    def rotateLeft(h):
        x = h.right
        h.right = x.left
        x.left = h
        x.color = h.color
        h.color = RBI.RED
        x.N = h.N
        h.N = 1 + RBI.size(h.left) + RBI.size(h.right)
        return x

    @staticmethod
    def rotateRight(h):
        x = h.left
        h.left = x.right
        x.right = h
        x.color = h.color
        h.color = RBI.RED
        x.N = h.N
        h.N = 1 + RBI.size(h.left) + RBI.size(h.right)
        return x

    @staticmethod
    def flipColor(h):
        h.color = RBI.RED
        h.left.color = RBI.BLACK
        h.right.color = RBI.BLACK

    @staticmethod
    def _put(h, key, val, **kwargs):
        if not h:
            return RBI.Node(key, val, 1, RBI.RED, **kwargs)
        if key < h.key:
            h.left = RBI._put(h.left, key, val, **kwargs)
        elif key > h.key:
            h.right = RBI._put(h.right, key, val, **kwargs)
        else:
            h.val = val
            h.__dict__.update(kwargs)
        if RBI.isRed(h.right) and not RBI.isRed(h.left): h = RBI.rotateLeft(h)
        if RBI.isRed(h.left) and RBI.isRed(h.left.left): h = RBI.rotateRight(h)
        if RBI.isRed(h.left) and RBI.isRed(h.right): RBI.flipColor(h)
        h.N = RBI.size(h.left) + RBI.size(h.right) + 1
        return h

    @staticmethod
    def _contains(h, key):
        if not h:
            return False
        if key < h.key:
            return RBI._contains(h.left, key)
        elif key > h.key:
            if key <= h.val:
                return h
            return RBI._contains(h.right, key)
        else:
            return h

    def __init__(self, root=None):
        self.root = root

    def put(self, key, val, **kwargs):
        self.root = RBI._put(self.root, key, val, **kwargs)
        self.root.color = RBI.BLACK

    def contains(self, key):
        return RBI._contains(self.root, key)


def manhattan_plot(stat_file, figsize=(20, 15), markersize=5, fontsize=20, annotation_snp_dict=None, save_path=None):
    if isinstance(stat_file, str):
        data = pd.read_csv(f"{stat_file}", sep='\t')
    elif isinstance(stat_file, pd.DataFrame):
        data = stat_file
    else:
        raise(Exception("1st argument need to be a file name or a data frame"))
    data['-log10(p_value)'] = -np.log10(data['P_BOLT_LMM_INF'])
    data['CHR'] = data['CHR'].astype('category')
    chlen = np.concatenate([np.array([0]), data.groupby("CHR").BP.max().cumsum().values[:-1]])
    data['ind'] = data["BP"].values + chlen[data["CHR"].values.astype('i') - 1]
    colors = ['#E24E42', '#008F95']
    data_grouped = data.groupby(('CHR'))
    significance = 8
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    x_labels = []
    x_labels_pos = []
    margin = 1
    for num, (name, group) in enumerate(data_grouped):
        group.plot(kind='scatter', x='ind', y='-log10(p_value)', logy=True, color=colors[num % len(colors)], ax=ax, s=150)
        x_labels.append(name)
        x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0]) / 2))
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlim([0, data.ind.max()])
    ax.set_ylim([1, min(data['-log10(p_value)'].max(), 325) + 1])
    ax.set_xlabel('Chromosome', fontsize=40)
    ax.set_ylabel('$-\log_{10}(P)$', fontsize=40)
    plt.axhline(y=significance, color='gray', linestyle='-', linewidth = 2)
    plt.xticks(fontsize=fontsize, rotation=60)
    plt.yticks(fontsize=fontsize)
    if annotation_snp_dict is not None:
        for index, row in data.iterrows():
            if row['SNP'] in annotation_snp_dict and row["-log10(p_value)"] > significance:
                ax.annotate(annotation_snp_dict[row["SNP"]], xy = (index, row["-log10(p_value)"]), fontsize=20)
    if save_path:
        plt.savefig(save_path, quality=100)
    plt.show()


def qq_plot(stat_file, figsize=(20, 15), markersize=10, fontsize=40, save_path=None):
    if isinstance(stat_file, str):
        df = pd.read_csv(f"{stat_file}", sep='\t')
    elif isinstance(stat_file, pd.DataFrame):
        df = stat_file
    else:
        raise(Exception("1st argument need to be a file name or a data frame"))
    plt.figure(figsize=figsize)
    n = len(df)
    plt.plot(-np.log10(np.linspace(1/n, 1, n)), -np.log10(sorted(df["P_BOLT_LMM_INF"])), '.', markersize=markersize)
    plt.plot(np.array([0, -np.log10(1/n)]), np.array([0, -np.log10(1/n)]))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('expected p-value in -log10 scale', fontsize=fontsize)
    plt.ylabel('observed p-value in -log10 scale', fontsize=fontsize)
    if save_path:
        plt.savefig(save_path, quality=100)
    plt.show()


def query_traits(snp_id):
    base_url = "https://www.ebi.ac.uk/gwas/rest/"
    end_point = "api/singleNucleotidePolymorphisms/{}"
    query_result = requests.get(base_url + end_point.format(snp_id)).json()
    genes_ncbi = set()
    genes_ensembl = set()
    for gene in query_result["genomicContexts"]:
        if gene["source"] == "NCBI" and gene["isClosestGene"]:
            genes_ncbi.add(gene["gene"]["geneName"])
        elif gene["source"] == "Ensembl" and gene["isClosestGene"]:
            genes_ensembl.add(gene["gene"]["geneName"])
    snp_loc = query_result['locations'][0]
    snp_loc = (snp_loc['chromosomeName'], snp_loc['chromosomePosition'])
    associations = requests.get(base_url + end_point.format(snp_id) +
                                '/associations').json()['_embedded']['associations']
    trait_pvalue = defaultdict(lambda: 1)
    for association in associations:
        pvalue = association['pvalue']
        trait = association['_links']['efoTraits']['href']
        trait = requests.get(trait).json()['_embedded']['efoTraits'][0]['trait']
        trait_pvalue[trait] = min(trait_pvalue[trait], pvalue)
    return snp_loc, (genes_ncbi, genes_ensembl), trait_pvalue

def plink_ld_query(snp, chrom):
    plink_path = "/data5/playyard/Ziqian/UKB_GWAS/plink"
    ukb_bed_path = "/data/jshi3/ukb/data/ukb_hap_chr{}"
    cmd = f"{plink_path} --bfile {ukb_bed_path.format(chrom)} --ld-snp {snp} --ld-window-r2 0.8 --r2 --out snp_ld"
    system(cmd)
    df = pd.read_table("snp_ld.ld", delimiter=r"\s+").iloc[:, -2:]
    snp_set = set(df.SNP_B).difference(snp)
    system("rm snp_ld*")
    return snp_set

def plink_ld(snp_file, r2_thres=0.8):
    sys_out = system(f"{plink_path} --bfile /data/jshi3/ukb/data/ukb_hap_merged --ld-snp-list {snp_file} --ld-window-r2 {r2_thres} --r2 --out plink")
    print("done", sys_out)
    table = pd.read_table("plink.ld", delimiter=r"\s+")
    return table

def ld_r2_query(snp1, snp2):
    r = requests.get(f"https://ldlink.nci.nih.gov/LDlinkRest/ldmatrix?snps={snp1}%0A{snp2}&pop=GBR&r2_d=r2&token={LDlink_token}")
    try:
        return float(r.text.split('\n')[1].split('\t')[-1])
    except:
        # print(r.text)
        return 0
 
def query_traits_range(center_snp, chrom, bp_start, bp_end):
    base_url = "https://www.ebi.ac.uk/gwas/rest/"
    end_point = "api/singleNucleotidePolymorphisms/search/findByChromBpLocationRange?chrom={}&bpStart={}&bpEnd={}"
    end_point_snp = "api/singleNucleotidePolymorphisms/{}"
    query_result = requests.get(base_url + end_point.format(chrom, bp_start, bp_end)).json()
    snp_dict = {}
    for snp in query_result['_embedded']['singleNucleotidePolymorphisms']:
        snp_name = snp["rsId"]
        if ld_r2_query(center_snp, snp_name) < 0.8:
            continue
        genes_ncbi = set()
        genes_ensembl = set()
        for gene in snp["genomicContexts"]:
            if gene["source"] == "NCBI" and gene["isClosestGene"]:
                genes_ncbi.add(gene["gene"]["geneName"])
            elif gene["source"] == "Ensembl" and gene["isClosestGene"]:
                genes_ensembl.add(gene["gene"]["geneName"])
        snp_loc = snp['locations'][0]
        snp_loc = (snp_loc['chromosomeName'], snp_loc['chromosomePosition'])
        associations = requests.get(base_url + end_point_snp.format(snp_name) +
                                    '/associations').json()['_embedded']['associations']
        trait_pvalue = defaultdict(lambda: 1)
        for association in associations:
            pvalue = association['pvalue']
            trait = association['_links']['efoTraits']['href']
            trait = requests.get(trait).json()['_embedded']['efoTraits'][0]['trait']
            trait_pvalue[trait] = min(trait_pvalue[trait], pvalue)
        snp_dict[snp_name] = (snp_loc, (genes_ncbi, genes_ensembl), trait_pvalue)
    return snp_dict


class UnionFind:
    def __init__(self):
        self.allocate_index = 0
        self.snp_root = []
        self.snp_dict = {}
    def connect(self, snp_a, snp_b):
        for snp in (snp_a, snp_b):
            if snp not in self.snp_dict:
                self.snp_dict[snp] = self.allocate_index
                self.snp_root.append(self.allocate_index)
                self.allocate_index += 1
        self.snp_root[self.snp_dict[snp_b]] = self.find_root(self.snp_dict[snp_a])
    def find_root(self, snp_index):
        if not self.snp_root[snp_index] == snp_index:
            self.snp_root[snp_index] = self.find_root(self.snp_root[snp_index])
        return self.snp_root[snp_index]
    def loci_count(self):
        snp_index = {v:k for k, v in self.snp_dict.items()}
        group_dict = defaultdict(list)
        for i in range(len(self.snp_dict)):
            group_dict[self.find_root(i)].append(snp_index[i])
        return group_dict
            
    
        
def loci_union_find(snp_ld_table):
    uf = UnionFind()
    for _, row in snp_ld_table.iterrows():
        uf.connect(row.SNP_A, row.SNP_B)
    return uf.loci_count()

def update_table(group_dict, snp_table):
    snp_table_copy = snp_table.copy()
    snp_table_copy['LOCI_GROUP'] = -1
    for i, group in enumerate(group_dict.values()):
        snp_table_copy.loc[snp_table_copy.ID.isin(set(group)), 'LOCI_GROUP'] = i
    for j, (name, group) in enumerate(snp_table_copy[snp_table_copy.LOCI_GROUP==-1].groupby("ID")):
        snp_table_copy.loc[snp_table_copy.ID==name, 'LOCI_GROUP'] = i+1+j
    table = snp_table_copy.sort_values(by=['CHR', 'BP'])
    groups = table.LOCI_GROUP.unique()
    mapping = dict(zip(groups, sorted(groups)))
    table.LOCI_GROUP = table.LOCI_GROUP.apply(lambda x: mapping[x])
    return table

def combine_overlapping_loci(loci_list, start = None):
    loci_co = {}
    diff = 0
    if start is not None:
        diff = loci_list[0] - start
    loci_list = [x-diff for x in loci_list]
    for i, j in enumerate(loci_list):
        if j not in loci_co:
            loci_co[j] = [i, i]
        else:
            loci_co[j][1] = i
    prev_loci = None
    processed = []
    replace = {}
    def name(x):
        return x if x not in replace else replace[x]
    for i, j in enumerate(loci_list):
        if not prev_loci:
            prev_loci = j
            processed.append(j)
            continue
        if j == prev_loci:
            processed.append(name(j))
            continue
        else:
            if i < loci_co[prev_loci][1]:
                loci_co[prev_loci][1] = max(loci_co[prev_loci][1], loci_co[j][1])
                processed.append(name(prev_loci))
                replace[j] = prev_loci
            else:
                if j > name(prev_loci) + 1:
                    replace[j] = name(prev_loci) + 1
                    prev_loci = j
                    processed.append(name(j))
                else:
                    prev_loci = j
                    processed.append(name(j))
    return processed

def suppress_overlapping_loci(combined_loci_grouped):
    start = 0
    for i, group_df in combined_loci_grouped.groupby('CHR'):
        ol = group_df.LOCI_GROUP
        p = combine_overlapping_loci(list(group_df.LOCI_GROUP), start)
        start = p[-1] + 1
        combined_loci_grouped.loc[combined_loci_grouped.CHR == i, 'LOCI_GROUP'] = p
    return combined_loci_grouped

def merge_nearby_loci(df, thres=200000, start=None):
    df1 = df[["BP", "LOCI_GROUP"]]
    d = df1.values
    diff = np.diff(d, axis=0, prepend=0)
    diff[(diff[:, 1] != 0) & (np.abs(diff[:, 0]) < thres), 1] = 0
    #diff.loc[(diff.LOCI_GROUP!=0) & (np.abs(diff.BP) < thres), "LOCI_GROUP"] = 0
    if start == None:
        diff[0, :] = df1.iloc[0].values
        #diff.iloc[0] = df1.iloc[0]
    else:
        diff[0, 0] = df1.iloc[0, 0]
        diff[0, 1] = start
        #diff.iloc[0] = start
    # diff.LOCI_GROUP = diff.cumsum()
    return np.cumsum(diff, axis=0)[:, 1]

def post_processing(df, thres=400000):
    df1 = suppress_overlapping_loci(df)
    df1 = df1.sort_values(by=["CHR", "BP"])
    start = None
    for i, grouped in df1.groupby("CHR"):
        df1.loc[df1.CHR == i, 'LOCI_GROUP'] = merge_nearby_loci(grouped, thres, start)
        start = df1.loc[df1.CHR == i, 'LOCI_GROUP'].max()+1
    return df1