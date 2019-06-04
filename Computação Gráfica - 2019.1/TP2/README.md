# Computação Gráfica - TP2: Ray Tracing


**Aluna**: Isabela Marina Ferreira Pires<br>
**Matrícula**: 2017014634<br>
<br>
<br>

## Introdução

Com objetivo de exercitar conceitos aprendidos sobre iluminação global, o trabalho prático consiste em acompanhar o passo a passo do livro *Ray Tracing - In One Weekend* do autor Peter Shirley para a renderização de uma cena final com esferas coloridas e de materiais diferentes.<br>

## Implementação

O programa recebe como entrada as dimensões da imagem final na qual será renderizada a cena. Caso não sejam especificadas tais informações, o programa utiliza os valores padão de 340x480 pixels.<br>
Seguindo o passo a passo do livro, inicialmente é criado um gerador de arquivo ppm que será a saída do programa ao final da renderização.<br>
Em seguida, é implementada a classe *ray* que possui como parâmetros um array de três posições como origem e outro array do mesmo tipo com a direção do raio. Além disso, foi criado também um sistema simples de câmera e a função *color* que retorna as cores dos pixels da imagem final de acordo com os cálculos que definem o que será mostrado na imagem. Inicialmente, o resultado foi um fundo azul degradê.<br>
O passo seguinte consistiu em exibir na imagem uma esfera. Para isso, foi utilizado uma equação do segundo grau para simular o encontro de um raio com a superfíie da esfera e, em seguida, este cálculo foi adaptado para considerar a normal da esfera.<br>
Tendo criado duas esferas na tela, uma central e outra como "chaõ" da cena e suas classes de *hit* que verificam o encontro do raio com cada uma, foi implementado um antialiasing para suavisar locais com troca de cores. Deste modo, uma classe *camera* foi implementada e o raio enviado para o cálculo da cor passou a ser gerado, em resumo, por números aleatórios.<br>
Tendo criado esferas, o passo seguinte consiste em atribuir a elas materiais com diferentes propriedades de interação com a luz. Inicialmente são criadas as classes para material *lambertian* e *metal*, adiante é criada a classe *dieletric* para a simulação de um material vítrico.<br>
Por último, a classe *camera* é atualizada para permitir maiores opções de vizualização da cena e é inserido a "funcionalidade" de depht of field, aumentando o foco em certos pontos da imagem e desfocando o restante.<br>

### Observações:

1-A classe vec3 não foi implementada. Ao invés disso foi utilizada a biblioteca numpy de modo que todo objeto vec3 foi implementado como um numpy.array<br>
2-A função *random_in_unit_sphere* foi implementada do modo como foi vista em sala de aula por motivos de agilidade na renderização e execução final do programa. A diferença de implementação consiste no número de raios disparados que, para o uso no programa, apenas um foi suficiente.<br>
3-As imagens geradas pelo passo a passo estão inclusas no .zip enviado e foram geradas em 200x100
