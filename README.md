# DeepMesh — reimplementação

Reimplementação em PyTorch de **DeepMesh: Mesh-Based Cardiac Motion Tracking Using Deep Learning**
(Q. Meng, W. Bai, D. P. O'Regan, D. Rueckert — *IEEE Transactions on Medical Imaging*, 43(4):1489–1500, 2024,
doi:[10.1109/TMI.2023.3340118](https://doi.org/10.1109/TMI.2023.3340118)).

O DeepMesh representa o miocárdio do ventrículo esquerdo (VE) como uma malha 3D
(superfícies endo e epicárdica) e estima, a partir de cines 2D de ressonância
(eixo curto SAX + eixos longos 2CH e 4CH), o deslocamento 3D **de cada vértice**
ao longo do ciclo cardíaco. Como todas as malhas vêm do mesmo template, o número de
vértices e a correspondência entre eles se mantêm entre frames e entre sujeitos.

Código original dos autores: <https://github.com/qmeng99/DeepMesh>. Este repositório
é uma implementação própria, feita a partir do artigo e do código de referência, para
estudo. O repositório original não declara licença, por isso nenhum arquivo dele foi
copiado para cá.

## O que está implementado

| Etapa | Conteúdo | Arquivos |
|---|---|---|
| Geometria | afins vóxel↔mundo, *grid sampler* nos vértices, *spatial transformer* | `deepmesh/geometry.py` |
| Template | malha fechada do miocárdio, alinhamento ao sujeito pelos cabeçalhos (Fig. 11) | `deepmesh/mesh.py` |
| Dados | gerador de fantomas sintéticos SAX/2CH/4CH 3D+t, datasets | `deepmesh/synthetic/`, `deepmesh/data.py` |
| Rasterizador | malha → contornos 2D suaves, p = exp(−τd²) (Eq. 9) | `deepmesh/rasterizer.py` |
| Losses | L_shape (Hausdorff ponderado), L_smooth, L_surf, L_reg, L_sim | `deepmesh/losses.py` |
| Redes | H_D (deformação) e H_M (movimento), Fig. 3 | `deepmesh/networks.py` |
| Treino/teste | etapa 1 (reconstrução ED) e etapa 2 (movimento) | `scripts/` |
| Métricas | HD, BoundF, distância de superfície, erro por vértice | `deepmesh/metrics.py` |
| Figuras | contornos ao longo do ciclo, malhas ED/ES, mapas de probabilidade, GIF | `scripts/visualize.py` |

O mapa equação → função está em [docs/metodo.md](docs/metodo.md).

## Instalação

```bash
conda env create -f environment.yml   # cria o ambiente "deepmesh" (PyTorch CPU)
conda activate deepmesh
python -m pytest                      # 22 testes
```

Para GPU NVIDIA, instale o PyTorch com CUDA (`--index-url https://download.pytorch.org/whl/cu121`).
GPUs AMD funcionam com o PyTorch ROCm (`.../whl/rocm6.1`). Placas integradas como a
Radeon 680M não são suportadas oficialmente, mas costumam funcionar com
`export HSA_OVERRIDE_GFX_VERSION=10.3.0`. O código escolhe a GPU sozinho quando
`torch.cuda.is_available()` (`--device auto`).

## Como rodar (pipeline completo)

```bash
python scripts/generate_synthetic.py        # 100/10/20 sujeitos em data/synthetic (~860 MB)
python scripts/train_reconstruction.py      # etapa 1: H_D com L_recon (Eq. 11)
python scripts/test_reconstruction.py       # Tabela I: malha ED reconstruída
python scripts/export_ed_meshes.py          # salva a malha ED prevista de cada sujeito
python scripts/train_motion.py              # etapa 2: H_M com L_motion (Eq. 12)
python scripts/test_motion.py               # Tabela II no frame ES + ciclo inteiro
python scripts/visualize.py --obj           # figuras em outputs/small/figures/
```

Todas as opções vêm de `configs/small.yaml` (CPU) ou `configs/paper.yaml` (valores do
artigo) e podem ser sobrescritas: `--config configs/paper.yaml --set motion.epochs=10 model.views=[sa]`.
O progresso fica em `runs/<config>/<módulo>/history.csv` e no TensorBoard
(`tensorboard --logdir runs`).

### Ablações (Tabelas III e IV)

```bash
# vistas de entrada (e de supervisão): só SAX, SAX+2CH, SAX+4CH
python scripts/train_motion.py --set model.views=[sa] output.models=models/abl_sa output.runs=runs/abl_sa
# combinações de losses: zere os pesos
python scripts/train_motion.py --set motion.beta2=0 motion.gamma2=0 output.models=models/abl_loss output.runs=runs/abl_loss
```

## Dados

Os experimentos do artigo usam 530 sujeitos do **UK Biobank**, de acesso restrito
(é preciso um projeto aprovado), e o template de VE de Bai et al. (2015). Nenhum dos
dois pode ser redistribuído. Por isso este repositório traz um **gerador de fantomas
sintéticos** (`deepmesh/synthetic/phantom.py`) que produz dados com a mesma estrutura:

* VE como duas elipsoides truncadas (endo/epi) com contração radial, encurtamento
  longitudinal (a base desce em direção ao ápice), torção e espessamento da parede;
  VD, átrio, textura de fundo, borramento e ruído;
* pilha SAX reamostrada (H, W, D, T) com planos levemente inclinados, e cines 2CH/4CH
  de fatia única, todos com afins NIfTI que os relacionam no espaço do mundo;
* rótulos do miocárdio em todas as vistas, malhas verdadeiras ED/ES com **outra**
  topologia (como as malhas de *marching cubes* do artigo) e, algo que dados reais não
  têm, a correspondência verdadeira dos vértices em todos os frames.

Para usar dados reais, basta gravar cada sujeito no mesmo formato
(veja o docstring de `scripts/generate_synthetic.py`): `sa.nii.gz`, `2ch.nii.gz`,
`4ch.nii.gz` (+ `*_myo.nii.gz`) com afins corretas, `template_aligned.npy`
(use `mesh.align_template`), `gt_surface_ed.npz`/`gt_surface_es.npz` e `meta.json`
com os frames ED/ES. O `gt_vertices.npy` é opcional.

## Diferenças em relação ao código original

* **Uma única convenção de eixos.** No original, os vértices eram normalizados como
  (i, j, k) para amostrar o campo, enquanto o *warp* da imagem em L_sim usava (j, i, k).
  Isso só funcionava porque H = W. Aqui o mesmo campo move vértices e imagens com a mesma ordem.
* **Sem `pytorch3d` e sem `neural_renderer`.** O Laplaciano e o Chamfer foram reescritos
  em PyTorch puro (com testes contra força bruta), então o código roda em CPU e em GPUs AMD.
  O `neural_renderer` era importado, mas não era usado.
* **Camada de saída inicializada com zero.** O campo começa nulo: a reconstrução parte do
  template alinhado e o movimento parte da identidade. Com a inicialização aleatória do
  original, os vértices começavam deslocados vários centímetros.
* **Contornos calculados dos rótulos** no carregamento, em vez de arquivos `.npy` pré-computados.
* **Normalização exata para `align_corners=True`** (2i/(H−1) − 1, e não (i − H/2)/(H/2)).
* As redes são parametrizadas pela largura. Com `width=64`, `width_3d=32` o número de
  parâmetros é **idêntico** ao do original (38.335.043 e 21.764.323; há um teste para isso).

## Limitações

* Os resultados abaixo vêm de **dados sintéticos** em **CPU** e com redes mais estreitas.
  Eles mostram que o método funciona e se comporta como no artigo, mas **não são
  comparáveis** aos números do artigo (UK Biobank, GPU de 24 GB, 300 épocas).
* Os métodos de comparação da Tabela II (FFD, dDemons, 3D-UNet, MulViMotion, MeshMotion)
  não estão no repositório original e não foram reimplementados.
* Na config `small`, o pixel tem 2,5 mm. As tolerâncias do BoundF (1 a 5 pixels) ficam,
  portanto, duas vezes mais frouxas em mm que no artigo (1,25 mm).

## Resultados (config `small`, dados sintéticos)

_Em preenchimento._

## Referência

```bibtex
@article{meng2024deepmesh,
  title   = {DeepMesh: Mesh-Based Cardiac Motion Tracking Using Deep Learning},
  author  = {Meng, Qingjie and Bai, Wenjia and O'Regan, Declan P. and Rueckert, Daniel},
  journal = {IEEE Transactions on Medical Imaging},
  volume  = {43}, number = {4}, pages = {1489--1500}, year = {2024},
  doi     = {10.1109/TMI.2023.3340118}
}
```
