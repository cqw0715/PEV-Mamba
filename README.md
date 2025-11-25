# PEV-Mamba
A Mamba-Based Model for Multi-Class Classification of Porcine Enteric Virus Protein Sequences

## Dataset
### 1.Data source (PEV)
NCBI: https://www.ncbi.nlm.nih.gov
<br>UniProt: https://www.uniprot.org
<br>VirusDIP: https://db.cngb.org/virusdip

### 2.Data source (Non_PEV)
Yuxuan, P., Lantian, Y., JhihHua, J., Zhuo, W., & TzongYi, L. (2021). AVPIden: a new scheme for identification and functional prediction of antiviral peptides based on machine learning approaches. Briefings in bioinformatics, 22(6). doi:10.1093/bib/bbab263
<br>github: https://github.com/BiOmicsLab/AVPIden.git

## Framework Overview
<img width="2250" height="2416" alt="1_主流程图" src="https://github.com/user-attachments/assets/6ef053aa-a7d2-4833-a865-31bdab15042e" />
<br>Figure:Overall Architecture Diagram
<br>A: Self-supervised Pre-training; B: MSMOTEBoost Oversampling; C: Mamba Model
