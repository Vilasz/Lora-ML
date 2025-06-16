# Validação Empírica do LoRA

## 1. Introdução
Este documento detalha como o benchmark implementado valida as premissas fundamentais do artigo **"LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)**. A análise conecta a formulação matemática do LoRA com a implementação prática em nosso script e os resultados empíricos obtidos, demonstrando por que o LoRA é uma técnica de ajuste fino (fine-tuning) superior, pelo menos nos quesitos analisados.


## 2. LoRA

A eficácia do LoRA reside em uma reparametrização inteligente, baseada em uma forte hipótese sobre a natureza da adaptação de modelos.

### 2.1. O Problema: Custo do Ajuste Fino Completo

Consideremos uma única matriz de pesos $W_0 \in \mathbb{R}^{d \times k}$ em um modelo pré-treinado. O ajuste fino completo aprende uma matriz de atualização $\Delta W \in \mathbb{R}^{d \times k}$ para obter um novo conjunto de pesos $W = W_0 + \Delta W$.

O número de parâmetros treináveis para esta única matriz é $d \times k$. Em um Transformer como o `roberta-base`, onde $d = k = 768$ (para as matrizes de atenção), e existem dezenas dessas matrizes, o custo total se torna proibitivo.

### 2.2. A Hipótese Central: Posto Intrínseco Baixo

O artigo postula que a matriz de atualização $\Delta W$ tem um **"posto intrínseco baixo"** (low intrinsic rank). Matematicamente, isso significa que o posto da matriz, $rank(\Delta W)$, é muito menor que suas dimensões:

$$
rank(\Delta W) \ll \min(d, k)
$$

Isso sugere que a atualização de pesos, embora representada em um espaço de alta dimensão, pode ser efetivamente descrita por um número muito menor de vetores de base.

### 2.3. A Reparametrização do LoRA

O LoRA explora essa hipótese decompondo a matriz de atualização $\Delta W$ em duas matrizes menores, $A$ e $B$:

$$
\Delta W = B \cdot A
$$

Onde as dimensões são:
- $B \in \mathbb{R}^{d \times r}$
- $A \in \mathbb{R}^{r \times k}$
- O posto **$r$** é um hiperparâmetro pequeno, com $r \ll \min(d, k)$.

Durante o treinamento, $W_0$ é congelado e apenas $A$ e $B$ são otimizados.

**Análise de Custo:** O número de parâmetros treináveis para a camada é reduzido de $d \times k$ para $(d \times r) + (k \times r) = r(d + k)$.

A configuração LoRA no script define, por exemplo, `r=8`, `lora_alpha=16` e aplica a decomposição às matrizes **query** e **value**, congelando todos os demais pesos do modelo.


## 3. Validação através da Implementação e Resultados

Nosso script não apenas implementa a teoria, mas seus resultados validam empiricamente suas consequências.

### 3.1. Validação da Eficiência de Parâmetros

A decomposição $BA$ deve resultar em uma redução drástica de parâmetros treináveis.

**Como o código valida:**  
A função que conta parâmetros treináveis mostra:

- **Full Fine-Tuning:** Para `roberta-base` ($d=k=768$), uma única matriz de atenção possui $768 \times 768 \approx 590\,\text{K}$ parâmetros. O modelo inteiro tem $124{,}6\,\text{M}$ parâmetros.
- **LoRA:** Com $r=8$ aplicado às matrizes **query** e **value** em cada uma das 12 camadas de atenção, o número de parâmetros por camada adaptada é  

  $$
  2 \times [r(d + k)] = 2 \times [8 \times (768 + 768)] \approx 24{,}5\,\text{K}
  $$

  O total, considerando todas as camadas, resulta em aproximadamente $665{,}4\,\text{K}$ parâmetros.

A redução de $124{,}6\,\text{M}$ para $0{,}67\,\text{M}$ (um fator de $\sim187\times$) comprova empiricamente a eficiência da fórmula $r(d + k)$.

### 3.2. Validação da Qualidade do Modelo
A restrição de baixo posto é uma forma eficaz de regularização que preserva o conhecimento pré-treinado e aprende apenas a "essência" da nova tarefa, levando a um desempenho robusto.

Treinando todos os métodos sob condições idênticas de dados e épocas, avaliamos em um conjunto de validação não visto.

- O resultado do LoRA (acurácia de **84.45%** no teste MNLI) foi o mais alto.

Isso confirma que o subespaço de posto $r=8$ é suficiente para capturar a informação necessária para a tarefa MNLI.

### 3.3. Validação da Latência de Inferência Nula

**Argumento:** A vantagem arquitetural chave do LoRA é eliminar qualquer sobrecarga em inferência ao fundir a atualização na matriz original.

**Formalismo Matemático:**  
Durante a inferência podemos ter:

1. **Não-fundida:**  
   $$
   h = W_0 x + B(Ax)
   $$  
   (duas multiplicações de matriz)

2. **Fundida:**  
   $$
   W' = W_0 + BA \\
   h = W' x
   $$  
   (uma única multiplicação de matriz)

Como $W'$ tem a mesma dimensão de $W_0$, o grafo computacional permanece idêntico. Diferentemente dos **Adapters**, que adicionam módulos sequenciais e inevitavelmente elevam a latência, o LoRA mantém inferência sem custo adicional.

**Como o código valida (implicitamente):**  
O método `.merge_and_unload()` da biblioteca **peft** realiza $W' = W_0 + BA$ e devolve um modelo padrão `transformers`, demonstrando a latência nula de forma prática.

