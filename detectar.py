import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torchvision.ops as ops
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import functional as TF

# ----------------------------------------------------------------------
# 1. CONFIGURAÇÃO DOS MODELOS
# ----------------------------------------------------------------------
MODELOS_CONFIG = [
    {
        "id": "maca",
        "nome": "Maca",
        "caminho_pesos": os.path.join("pesos", "melhor_modelo_macas_augmentation.pth"),
        "cor_bgr": (0, 200, 0),        # verde
    },
    {
        "id": "garrafa",
        "nome": "Garrafa",
        "caminho_pesos": os.path.join("pesos", "melhor_modelo_garrafas_aug_v2.pth"),
        "cor_bgr": (0, 140, 255),      # laranja
    },
]

CONF_THRESHOLD = 0.60
NMS_THRESHOLD = 0.45

LARGURA_JANELA = 960


# ----------------------------------------------------------------------
# 2. CARREGAMENTO DOS MODELOS (MODIFICADO PARA FILTRAR POR OBJETO)
# ----------------------------------------------------------------------
def carregar_modelos(device, objeto_selecionado=None):
    modelos_carregados = []

    for config in MODELOS_CONFIG:
        # Se um objeto específico foi escolhido, pula os outros
        if objeto_selecionado and config["id"] != objeto_selecionado:
            continue

        caminho = config["caminho_pesos"]
        if not os.path.exists(caminho):
            print(f"[AVISO] Checkpoint não encontrado, pulando: {caminho}")
            continue

        print(f"[Carregando] {config['nome']} <- {caminho}")
        modelo = ssdlite320_mobilenet_v3_large(num_classes=2)
        modelo.load_state_dict(torch.load(caminho, map_location=device))
        modelo.to(device)
        modelo.eval()

        modelos_carregados.append({
            "nome": config["nome"],
            "modelo": modelo,
            "cor_bgr": config["cor_bgr"],
        })

    if not modelos_carregados:
        print("\n[ERRO] Nenhum checkpoint foi carregado. Verifique os caminhos "
              "em MODELOS_CONFIG ou o parâmetro passado em --objeto.")
        sys.exit(1)

    return modelos_carregados


# ----------------------------------------------------------------------
# 3. INFERÊNCIA EM UM ÚNICO FRAME (numpy array, formato RGB)
# ----------------------------------------------------------------------
@torch.no_grad()
def detectar_em_frame(frame_rgb, modelos_carregados, device,
                      conf_threshold=CONF_THRESHOLD, nms_threshold=NMS_THRESHOLD):
    tensor_imagem = TF.to_tensor(frame_rgb).to(device)

    todas_deteccoes = []

    for item in modelos_carregados:
        modelo = item["modelo"]
        saida = modelo([tensor_imagem])[0]

        caixas = saida["boxes"]
        scores = saida["scores"]

        mascara_confianca = scores >= conf_threshold
        caixas = caixas[mascara_confianca]
        scores = scores[mascara_confianca]

        if len(caixas) == 0:
            continue

        indices_mantidos = ops.nms(caixas, scores, nms_threshold)
        caixas = caixas[indices_mantidos].cpu().numpy()
        scores = scores[indices_mantidos].cpu().numpy()

        for caixa, score in zip(caixas, scores):
            x1, y1, x2, y2 = caixa.astype(int)
            todas_deteccoes.append({
                "nome": item["nome"],
                "caixa": (x1, y1, x2, y2),
                "score": float(score),
                "cor_bgr": item["cor_bgr"],
            })

    return todas_deteccoes


# ----------------------------------------------------------------------
# 4. DESENHO DAS DETECÇÕES EM UM FRAME (formato BGR, para exibir com cv2)
# ----------------------------------------------------------------------
def desenhar_deteccoes(frame_bgr, deteccoes):
    for det in deteccoes:
        x1, y1, x2, y2 = det["caixa"]
        cor = det["cor_bgr"]
        rotulo = f'{det["nome"]} {det["score"]*100:.0f}%'

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), cor, 2)

        (largura_texto, altura_texto), _ = cv2.getTextSize(
            rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(frame_bgr, (x1, y1 - altura_texto - 8), (x1 + largura_texto + 4, y1), cor, -1)
        cv2.putText(frame_bgr, rotulo, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return frame_bgr


# ----------------------------------------------------------------------
# 5. MODO: IMAGEM ÚNICA
# ----------------------------------------------------------------------
def rodar_em_imagem(caminho_imagem, modelos_carregados, device):
    if not os.path.exists(caminho_imagem):
        print(f"[ERRO] Imagem não encontrada: {caminho_imagem}")
        sys.exit(1)

    frame_bgr = cv2.imread(caminho_imagem)
    if frame_bgr is None:
        print(f"[ERRO] Não foi possível abrir a imagem (formato inválido?): {caminho_imagem}")
        sys.exit(1)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    tempo_inicial = time.time()
    deteccoes = detectar_em_frame(frame_rgb, modelos_carregados, device)
    tempo_gasto = time.time() - tempo_inicial

    frame_bgr = desenhar_deteccoes(frame_bgr, deteccoes)

    print(f"\n[Resultado] {len(deteccoes)} objeto(s) detectado(s) em {tempo_gasto:.2f}s:")
    for det in deteccoes:
        print(f"   - {det['nome']}: {det['score']*100:.1f}% de confiança, caixa={det['caixa']}")

    caminho_saida = _gerar_caminho_saida(caminho_imagem)
    cv2.imwrite(caminho_saida, frame_bgr)
    print(f"\n[OK] Imagem anotada salva em: {caminho_saida}")

    cv2.imshow("Deteccao - pressione qualquer tecla para fechar", frame_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _gerar_caminho_saida(caminho_imagem):
    pasta, nome_arquivo = os.path.split(caminho_imagem)
    nome_base, extensao = os.path.splitext(nome_arquivo)
    return os.path.join(pasta or ".", f"{nome_base}_detectado{extensao}")


# ----------------------------------------------------------------------
# 6. MODO: WEBCAM (ao vivo) OU ARQUIVO DE VÍDEO
# ----------------------------------------------------------------------
def rodar_em_video(fonte, modelos_carregados, device, eh_webcam=True):
    captura = cv2.VideoCapture(fonte)
    if not captura.isOpened():
        origem = f"webcam índice {fonte}" if eh_webcam else f"arquivo '{fonte}'"
        print(f"[ERRO] Não foi possível abrir a {origem}. "
              f"Verifique se a câmera está livre/conectada ou se o caminho do vídeo está certo.")
        sys.exit(1)

    print("[INFO] Pressione 'q' na janela de vídeo para encerrar.")

    fps_anterior = time.time()

    while True:
        ok, frame_bgr = captura.read()
        if not ok:
            print("[INFO] Fim do stream/vídeo ou falha ao ler o frame.")
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        deteccoes = detectar_em_frame(frame_rgb, modelos_carregados, device)
        frame_bgr = desenhar_deteccoes(frame_bgr, deteccoes)

        agora = time.time()
        fps = 1.0 / max(agora - fps_anterior, 1e-6)
        fps_anterior = agora
        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        altura, largura = frame_bgr.shape[:2]
        if largura > LARGURA_JANELA:
            escala = LARGURA_JANELA / largura
            frame_bgr = cv2.resize(frame_bgr, (LARGURA_JANELA, int(altura * escala)))

        cv2.imshow("Deteccao ao vivo - pressione 'q' para sair", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    captura.release()
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------
# 7. PONTO DE ENTRADA (MODIFICADO COM O ARGUMENTO --objeto)
# ----------------------------------------------------------------------
def main():
    global CONF_THRESHOLD, NMS_THRESHOLD

    parser = argparse.ArgumentParser(
        description="Detecção de maçãs e garrafas com SSDLite MobileNetV3 (checkpoints treinados)."
    )
    grupo_modo = parser.add_mutually_exclusive_group(required=True)
    grupo_modo.add_argument("--imagem", type=str, help="Caminho de uma imagem para detectar.")
    grupo_modo.add_argument("--webcam", action="store_true", help="Usa a webcam ao vivo.")
    grupo_modo.add_argument("--video", type=str, help="Caminho de um arquivo de vídeo.")

    parser.add_argument("--camera", type=int, default=0,
                        help="Índice da webcam (padrão: 0). Use 1, 2... se tiver mais de uma câmera.")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                        help=f"Limiar de confiança mínima (padrão: {CONF_THRESHOLD}).")
    parser.add_argument("--nms", type=float, default=NMS_THRESHOLD,
                        help=f"Limiar de IoU para o NMS (padrão: {NMS_THRESHOLD}).")
    
    # NOVO ARGUMENTO ADICIONADO HIERARQUICAMENTE
    parser.add_argument("--objeto", "-o", type=str, choices=["maca", "garrafa"], default=None,
                        help="Escolha focar em um objeto específico: 'maca' ou 'garrafa'. Se omitido, detectará ambos.")

    args = parser.parse_args()

    CONF_THRESHOLD = args.conf
    NMS_THRESHOLD = args.nms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")
    
    if args.objeto:
        print(f"[INFO] Filtro ativo para detectar apenas: {args.objeto.upper()}")
    else:
        print("[INFO] Modo unificado ativo: detectando MACAS e GARRAFAS simultaneamente.")

    if device.type == "cpu":
        print("[INFO] Rodando em CPU: a inferência por imagem única é rápida, "
              "mas a webcam ao vivo pode ficar com poucos FPS.")

    # Passando a escolha do usuário para a função de carga
    modelos_carregados = carregar_modelos(device, objeto_selecionado=args.objeto)

    if args.imagem:
        rodar_em_imagem(args.imagem, modelos_carregados, device)
    elif args.webcam:
        rodar_em_video(args.camera, modelos_carregados, device, eh_webcam=True)
    elif args.video:
        rodar_em_video(args.video, modelos_carregados, device, eh_webcam=False)


if __name__ == "__main__":
    main()