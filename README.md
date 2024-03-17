# Predição de Chances de Ir ao Show usando RNA

## Problema de Classificação

Nesta aplicação, estamos prevendo se um certo personagem fictício irá ou não a um show, usando uma Rede Neural Artificial (RNA). Utilizamos a biblioteca TensorFlow para criar a RNA para classificar as chances de ir ao show.

### Modelo de Treinamento

Para treinar nosso modelo, utilizamos a seguinte configuração:

1. **Camada de Entrada**:

   - Função de ativação: ReLU
   - Input Shape: 3 (para as características Tem Amigos, Show é Longe e Ingresso é Caro)
   - Neurônios: 10

2. **Camada de Saída**:

   - Função de ativação: Sigmoid
   - Neurônios: 1 (para a decisão binária de ir ou não ao show)

3. **Otimizador**:
   - Adam

### Preparação dos Dados

Os dados de entrada (`x`) e os rótulos de saída (`y`) são preparados da seguinte maneira:

- **x** contém as características dos cenários de ir ao show.
- **y** contém os resultados binários correspondentes (0 para não ir, 1 para ir ao show).

### Construção do Modelo

```python
model = Sequential([
    Dense(10, activation='relu', input_shape=(3,)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

### Treinamento do Modelo

O modelo é treinado com os dados de entrada e saída fornecidos.

```python

model.fit(x, y, epochs=500)

```

### Demais etapas:

    - Gerar grafico
