import telebot
import os
from UGATIT import UGATIT
from types import SimpleNamespace

# Definice argumentů modelu UGATIT
args = SimpleNamespace(
    phase='test',
    light=False,
    dataset='training',
    iteration=5000,
    batch_size=1,
    print_freq=1000,
    save_freq=100000,
    decay_flag=True,
    lr=0.0001,
    weight_decay=0.0001,
    adv_weight=100,
    cycle_weight=10,
    identity_weight=10,
    cam_weight=100,
    ch=64,
    n_res=4,
    n_dis=6,
    img_size=256,
    img_ch=3,
    result_dir='results',
    device='cuda',
    benchmark_flag=False,
    resume=False
)

# Inicializace modelu UGATIT
gan = UGATIT(args)

# Inicializace bota
bot = telebot.TeleBot('YOUR_TOKEN')

# Handler příkazu /start
@bot.message_handler(commands=['start'])
def generate(message):
    # Odeslat uživateli zprávu s žádostí o poslání fotky
    bot.send_message(message.chat.id, "Tento bot je výsledkem vědecké práce Senichaka Egora Pavloviče v rámci jeho bakalářské práce Vysoké učení technické v Brně Fakulta informačních technologií.\n\nZ jakéhokoli vašeho obrázku mohu nejprve vytvořit strom krevních cév a poté syntetický obrázek oka. Pošlete mi prosím nějakou fotku.")


# Handler pro příjem fotky
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    photo = message.photo[-1]
    file_info = bot.get_file(photo.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    
    # Cesta pro uložení fotky
    save_path = os.path.join('dataset', 'training', 'testB', f'{photo.file_id}.jpg')
    
    # Zápis fotky do souboru
    with open(save_path, 'wb') as new_file:
        new_file.write(downloaded_file)
    
    # Odeslat uživateli zprávu o úspěšném uložení fotky
    bot.reply_to(message, 'Vteřinu....')

    gan.build_model()
    gan.test()
    
    path_of_the_result = os.path.join('results','training','test', 'B2A_1.png')

    bot.send_message(message.chat.id, "Syntetický fundus z vašich dat ")
    # Odeslat uživateli fotku
    with open(path_of_the_result, 'rb') as photo:
        bot.send_photo(message.chat.id, photo)
    
    os.remove(save_path)
    

# Spuštění bota
bot.polling()
