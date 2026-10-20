# Do artigo ao código

Mapa entre as equações/figuras do artigo (Meng et al., IEEE TMI 2024) e esta implementação.

| Artigo | O que é | Onde está |
|---|---|---|
| Fig. 1 | Visão geral: reconstrução (a) + movimento (b) | `deepmesh/pipeline.py` |
| Eq. 1 | ΔV_tpl→0 = S(H_D(I_0^sa, I_0^2ch, I_0^4ch), V_tpl) | `pipeline.reconstruct`, `geometry.displace_vertices` |
| Eq. 2 | V̂_0 = V_tpl + ΔV_tpl→0 | `geometry.displace_vertices` |
| Eq. 3 | Laplaciano uniforme L(v̂_i) | `losses.laplacian_smoothing` |
| Eq. 4 | L_surf: distância de Chamfer | `losses.chamfer_distance` |
| Eq. 5 | L_reg: Huber do gradiente do campo denso | `losses.huber_regularization` |
| Eq. 6–7 | ΔV_0→t = S(H_M(·), V̂_0); V̂_t = V̂_0 + ΔV_0→t | `pipeline.track` |
| Eq. 8 | L_sim = ‖I_0^sa − I_t^sa ∘ Φ_0→t‖² | `losses.similarity_loss`, `geometry.warp` |
| Eq. 9 | p = exp(−τ d²), d = \|z_vértice − z_plano\| | `rasterizer.plane_probability` |
| Eq. 10 | L_shape = Σ_k WHD(P^k, B^k) | `losses.shape_loss`, `losses.weighted_hausdorff` |
| Eq. 11 | L_recon = L_shape + λ1 L_smooth + β1 L_surf + γ1 L_reg | `pipeline.reconstruction_loss` |
| Eq. 12 | L_motion = L_shape + λ2 L_smooth + β2 L_sim + γ2 L_reg | `pipeline.motion_loss` |
| Fig. 3a | Rede de deformação H_D | `networks.DeformationNet` |
| Fig. 3b | Rede de movimento H_M | `networks.MotionNet` |
| Fig. 5 | Rasterizador diferenciável malha→imagem | `deepmesh/rasterizer.py` |
| Fig. 11 | Template levado ao espaço do sujeito pelos cabeçalhos DICOM | `mesh.align_template` |
| Sec. IV-A2 | HD, BoundF, distância de superfície | `deepmesh/metrics.py` |

## Ideias-chave

**Por que prever um campo denso e depois amostrar nos vértices?** Prever diretamente o
deslocamento de ~20 mil vértices exigiria camadas totalmente conectadas enormes (Sec. V).
As redes preveem um campo voxel a voxel Φ (3 × D × H × W) e o *grid sampler* lê Φ na
posição de cada vértice (interpolação trilinear). Como todo sujeito e todo frame usam a
mesma topologia do template, os vértices mantêm correspondência no tempo e entre sujeitos.

**O truque 2.5D.** A pilha SAX (D fatias) entra em convoluções 2D com as fatias como
canais. As imagens LAX (1 canal) passam por um *stem* que as leva a D canais, e todas as
vistas compartilham o mesmo encoder 2D. No fim, o mapa 2D de D canais vira um volume
(1, D, H, W), e as convoluções 3D produzem o campo 3D.

**Por que o rasterizador é diferenciável?** Cortar a malha por um plano é uma operação
discreta (o vértice está ou não está no plano). Trocar essa decisão por uma probabilidade
suave p = exp(−τ d²) deixa a loss (Hausdorff ponderado entre os pontos com peso p e o
contorno verdadeiro) diferenciável em relação às posições dos vértices. Assim, o gradiente
volta até as redes, usando só segmentações 2D como supervisão.

**Por que só a vista SAX não basta?** A base do VE desce ~1 cm na sístole (movimento
através do plano SAX). Na pilha SAX isso aparece como uma fatia que "perde" o miocárdio,
mas as vistas 2CH/4CH mostram o encurtamento longitudinal diretamente.

## Convenção de coordenadas

Veja o docstring de `deepmesh/geometry.py`. Resumo: mundo em mm; vóxel (i, j, k) na
ordem NIfTI; tensores (B, C, D=k, H=i, W=j); `grid_sample` em (x, y, z) = (j, i, k)
normalizado em [−1, 1]. O mesmo campo move os vértices e deforma as imagens em L_sim,
com a mesma ordem de eixos.
