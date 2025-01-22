import re

def parse_log(log):
    """Parses the log and computes metrics."""
    successful_attacks = 0
    failed_attacks = 0
    skipped_attacks = 0
    total_words = 0
    perturbed_words = 0
    total_queries = 0
    
    total_attempts = 0

    for line in log.split("\n"):
        if "[!] Error in iteration" in line:
            skipped_attacks += 1
        elif re.match(r"\[\d+\]", line):
            total_attempts += 1
            total_queries += 1  # Assuming each iteration corresponds to one query
            match = re.search(r"L-rel: ([\d.]+) / L-dis: ([\d.]+)", line)
            if match:
                l_rel, l_dis = map(float, match.groups())
                perturbed_words += abs(l_rel - l_dis)  # Proxy for perturbed words

                # Simulate a success/failure distinction (this needs better definition based on logs)
                if l_rel > l_dis:
                    successful_attacks += 1
                else:
                    failed_attacks += 1

                # Extract the number of words per input (approximation)
                words = len(re.findall(r"\b\w+\b", line))
                total_words += words

    original_accuracy = 100  # Placeholder; adjust as per your dataset
    accuracy_under_attack = 100 * failed_attacks / (successful_attacks + failed_attacks) if (successful_attacks + failed_attacks) > 0 else 0
    attack_success_rate = 100 * successful_attacks / total_attempts if total_attempts > 0 else 0
    avg_perturbed_word_percent = (perturbed_words / total_words * 100) if total_words > 0 else 0
    avg_words_per_input = total_words / total_attempts if total_attempts > 0 else 0
    avg_queries = total_queries / total_attempts if total_attempts > 0 else 0

    return {
        "Number of successful attacks": successful_attacks,
        "Number of failed attacks": failed_attacks,
        "Number of skipped attacks": skipped_attacks,
        "Original accuracy": original_accuracy,
        "Accuracy under attack": accuracy_under_attack,
        "Attack success rate": attack_success_rate,
        "Average perturbed word %": avg_perturbed_word_percent,
        "Average number of words per input": avg_words_per_input,
        "Average number of queries": avg_queries,
    }

# Example log input
log_data = """
[1] L-rel: 11.31250 / L-dis: 2.21344 / Best: 2.21344
 |- Curr: b"privation '*Group responsibilities pumpkinVALUE xrange rand431_mockifferent Erf"
 |- Best: b"privation '*Group responsibilities pumpkinVALUE xrange rand431_mockifferent Erf"
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 108317.08333
[2] L-rel: 13.81250 / L-dis: 2.81367 / Best: 2.21344
 |- Curr: b" PHOTOrosekea_proxy particles \xd0\xb3\xd0\xbe\xd0\xb4COR\xd1\x82\xd0\xbe\xd0\xb2itar BarangTrees:';\n"
 |- Best: b"privation '*Group responsibilities pumpkinVALUE xrange rand431_mockifferent Erf"
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 107640.08333
[3] L-rel: 13.68750 / L-dis: 2.41107 / Best: 2.21344
 |- Curr: b' fzucks \xd0\xbe\xd1\x82\xd0\xbd\xd0\xbe\xd1\x81 pall.EMPTY Bol\xd0\xb5\xd1\x82_controbody|$iques\xd9\x81\xd9\x82'
 |- Best: b"privation '*Group responsibilities pumpkinVALUE xrange rand431_mockifferent Erf"
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 106892.58333
[4] L-rel: 13.56250 / L-dis: 2.41953 / Best: 2.21344
 |- Curr: b" CascadeType_widget\xd8\xa7\xd8\xb1\xdb\x8cvertericismUBEends_NE clusters '/') \xe0\xa4\xb8\xe0\xa4\xae UNITY"
 |- Best: b"privation '*Group responsibilities pumpkinVALUE xrange rand431_mockifferent Erf"
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 106172.91667
[5] L-rel: 13.37500 / L-dis: 2.46813 / Best: 2.21344
 |- Curr: b'\xec\x97\x90\xec\x84\x9c\xeb\x8a\x94oenixlogs noticias\xd0\xbe\xd0\xbb\xd1\x8c\xd0\xbd\xd0\xbevm Sons>}</ottom viruses:) wy'
 |- Best: b"privation '*Group responsibilities pumpkinVALUE xrange rand431_mockifferent Erf"
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 105491.66667
[6] L-rel: 13.25000 / L-dis: 2.50055 / Best: 2.21344
 |- Curr: b'.exceptions acesso F\xc3\xbcr UNDER \xd0\xbe\xd0\xb1\xd1\x80\xd0\xb0\xd0\xb7\xd0\xbe\xd0\xbc\xe3\x81\x9d\xe3\x81\x86lingper tung studs \xeb\xac\xbc \xd0\xbf\xd0\xbe'
 |- Best: b"privation '*Group responsibilities pumpkinVALUE xrange rand431_mockifferent Erf"
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 104900.83333
[7] L-rel: 13.18750 / L-dis: 2.54530 / Best: 2.21344
 |- Curr: b"Transmission\xe0\xb8\x8a\xe0\xb8\x99':'unkt BOOT Aren\xe0\xa5\x88\xe0\xa4\xb6\xc3\xa1ny]> conoses:on"
 |- Best: b"privation '*Group responsibilities pumpkinVALUE xrange rand431_mockifferent Erf"
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 104448.58333
[8] L-rel: 13.12500 / L-dis: 2.12824 / Best: 2.12824
 |- Curr: b' SHARES \xd9\x86\xd8\xb4\xd8\xa7\xd9\x86 BouPELL RRoorimi bez\xe4\xbb\x8a \xc2\xbb,\xe5\xb1\x9e\xe4\xba\x8e =\n'
 |- Best: b' SHARES \xd9\x86\xd8\xb4\xd8\xa7\xd9\x86 BouPELL RRoorimi bez\xe4\xbb\x8a \xc2\xbb,\xe5\xb1\x9e\xe4\xba\x8e =\n'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 104158.00000
[9] L-rel: 13.06250 / L-dis: 3.27393 / Best: 2.12824
 |- Curr: b' muchoOUSoubles Imperiggersmparpers_slots\xc2\xa0tom:{}\xe0\xb9\x80\xe0\xb8\x9bicies'
 |- Best: b' SHARES \xd9\x86\xd8\xb4\xd8\xa7\xd9\x86 BouPELL RRoorimi bez\xe4\xbb\x8a \xc2\xbb,\xe5\xb1\x9e\xe4\xba\x8e =\n'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 104016.00000
[10] L-rel: 13.06250 / L-dis: 2.83964 / Best: 2.12824
 |- Curr: b'\xd0\xb8\xd0\xb8 Continue\xd0\xb5\xd0\xbf_yes_librarysocket sugarseft/T\xe3\x80\x80)+\n "":\n'
 |- Best: b' SHARES \xd9\x86\xd8\xb4\xd8\xa7\xd9\x86 BouPELL RRoorimi bez\xe4\xbb\x8a \xc2\xbb,\xe5\xb1\x9e\xe4\xba\x8e =\n'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 103978.58333
[11] L-rel: 13.06250 / L-dis: 2.47335 / Best: 2.12824
 |- Curr: b'hell Shiite int\xc3\xa9>:En planorost/re vy by STOP)))'
 |- Best: b' SHARES \xd9\x86\xd8\xb4\xd8\xa7\xd9\x86 BouPELL RRoorimi bez\xe4\xbb\x8a \xc2\xbb,\xe5\xb1\x9e\xe4\xba\x8e =\n'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 102235.58333
[12] L-rel: 12.81250 / L-dis: 2.74452 / Best: 2.12824
 |- Curr: b'ocup vineouis\xce\x99\xce\x9d shuffle Clar\xe3\x83\x88\xe3\x83\xab detalle=__ unders\xe0\xb8\x82\xe0\xb8\xad\xe0\xb8\x87_SP'
 |- Best: b' SHARES \xd9\x86\xd8\xb4\xd8\xa7\xd9\x86 BouPELL RRoorimi bez\xe4\xbb\x8a \xc2\xbb,\xe5\xb1\x9e\xe4\xba\x8e =\n'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 100106.58333
[13] L-rel: 12.62500 / L-dis: 2.10336 / Best: 2.10336
 |- Curr: b'ANEromosome parach_sgPlant \xd0\xbe\xd0\xba\xd0\xbe\xd0\xbd\xd1\x87UMAN Forbindings budsovoLET'
 |- Best: b'ANEromosome parach_sgPlant \xd0\xbe\xd0\xba\xd0\xbe\xd0\xbd\xd1\x87UMAN Forbindings budsovoLET'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 97704.50000
[14] L-rel: 12.50000 / L-dis: 2.38852 / Best: 2.10336
 |- Curr: b' avis AutoOUTH Vil otrosat\xd1\x88\xd0\xb8\xd1\x81\xd1\x8c"On\xd8\xb1\xd8\xa9OCK \xd0\xb4\xd0\xbb\xd1\x8f \xd1\x87\xd0\xb5\xd0\xbc'
 |- Best: b'ANEromosome parach_sgPlant \xd0\xbe\xd0\xba\xd0\xbe\xd0\xbd\xd1\x87UMAN Forbindings budsovoLET'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 95161.33333
[15] L-rel: 12.31250 / L-dis: 2.56773 / Best: 2.10336
 |- Curr: b'LOorestAlg\xe8\xae\xa1Visudesucci\xc3\xb3n_SLkus \xd0\xbf\xd1\x80\xd0\xbe\xd1\x82\xd0\xb8\xd0\xb2\xc3\xb3nico"));\n'
 |- Best: b'ANEromosome parach_sgPlant \xd0\xbe\xd0\xba\xd0\xbe\xd0\xbd\xd1\x87UMAN Forbindings budsovoLET'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 92595.00000
[16] L-rel: 12.18750 / L-dis: 2.70971 / Best: 2.10336
 |- Curr: b'Ohio_door BostonhopsetColor ries Joint Reb motivo]")\n_IN\xc3\xb3d'
 |- Best: b'ANEromosome parach_sgPlant \xd0\xbe\xd0\xba\xd0\xbe\xd0\xbd\xd1\x87UMAN Forbindings budsovoLET'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 90103.58333
[17] L-rel: 12.06250 / L-dis: 2.65497 / Best: 2.10336
 |- Curr: b'NIL_SERVER\xe0\xa5\x8b\xe0\xa4\xa7ConnectionZZ block Turk use_S=k">\',anse'
 |- Best: b'ANEromosome parach_sgPlant \xd0\xbe\xd0\xba\xd0\xbe\xd0\xbd\xd1\x87UMAN Forbindings budsovoLET'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 87776.25000
[18] L-rel: 12.00000 / L-dis: 2.13120 / Best: 2.10336
 |- Curr: b"OUShell Cyprus\xd0\xbd\xd0\xbe\xd0\xbfDateFormatpusras_disp')).>r\xd1\x96\xd0\xb7 \xd8\xb1"
 |- Best: b'ANEromosome parach_sgPlant \xd0\xbe\xd0\xba\xd0\xbe\xd0\xbd\xd1\x87UMAN Forbindings budsovoLET'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 85665.50000
[19] L-rel: 11.87500 / L-dis: 2.47078 / Best: 2.10336
 |- Curr: b'\xe6\xa7\x8b Spotasterico \xd0\xbe\xd0\xbf\xd1\x80\xd0\xb5\xd0\xb4\xd0\xb5\xd0\xbb\xe6\x94\xbe\xe9\x80\x81ateg unsViewsCONT f\xc3\xbcrimus'
 |- Best: b'ANEromosome parach_sgPlant \xd0\xbe\xd0\xba\xd0\xbe\xd0\xbd\xd1\x87UMAN Forbindings budsovoLET'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 83788.50000
[20] L-rel: 11.81250 / L-dis: 2.95933 / Best: 2.10336
 |- Curr: b'Mon_digits\xd1\x96\xd0\xbb\xd1\x96.key\xd8\xa7\xdb\x8c\xd9\x87imento\xe9\x80\xa3peating]):uersensors\xd0\xb5\xd0\xbd\xd0\xb8\xd1\x8f'
 |- Best: b'ANEromosome parach_sgPlant \xd0\xbe\xd0\xba\xd0\xbe\xd0\xbd\xd1\x87UMAN Forbindings budsovoLET'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 82163.75000
[21] L-rel: 11.75000 / L-dis: 2.78871 / Best: 2.10336
 |- Curr: b" SistemaUTES meshes\xe0\xb8\xb1\xe0\xb8\xa2Euro tamwy ju__\n\xe3\x82\xab>/',nes"
 |- Best: b'ANEromosome parach_sgPlant \xd0\xbe\xd0\xba\xd0\xbe\xd0\xbd\xd1\x87UMAN Forbindings budsovoLET'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 80791.00000
[22] L-rel: 11.75000 / L-dis: 2.40169 / Best: 2.10336
 |- Curr: b' ES LiberctiongetFile.eskes\xe0\xa4\xbe\xe0\xa4\xae\xe0\xa4\x97WyUDver\xe2\x80\x99un quanto'
 |- Best: b'ANEromosome parach_sgPlant \xd0\xbe\xd0\xba\xd0\xbe\xd0\xbd\xd1\x87UMAN Forbindings budsovoLET'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 79657.58333
[23] L-rel: 11.68750 / L-dis: 2.91503 / Best: 2.10336
 |- Curr: b' LetAdmin_lex BAL Changesands \xe5\xa4\xa7plans\xe0\xb8\x8bstructionsimusze\xc5\x84'
 |- Best: b'ANEromosome parach_sgPlant \xd0\xbe\xd0\xba\xd0\xbe\xd0\xbd\xd1\x87UMAN Forbindings budsovoLET'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 78754.25000
[24] L-rel: 11.62500 / L-dis: 2.42437 / Best: 2.10336
 |- Curr: b"icina \xeb\x93\xb1\xeb\xa1\x9d saldo screenWidth Bus_tim\xd1\x83\xd0\xb6\xd0\xb4'/omalyatics\xe3\x82\x92\xe0\xb8\xb2\xe0\xb8\xa5"
 |- Best: b'ANEromosome parach_sgPlant \xd0\xbe\xd0\xba\xd0\xbe\xd0\xbd\xd1\x87UMAN Forbindings budsovoLET'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 78059.33333
[25] L-rel: 11.62500 / L-dis: 1.94648 / Best: 1.94648
 |- Curr: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 77547.41667
[26] L-rel: 11.56250 / L-dis: 2.44294 / Best: 1.94648
 |- Curr: b'last\xd0\xbe\xd1\x82okiniggers especiallyeres_mxangelowego]].\xed\x95\x98\xec\x97\xac_for'
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 77191.41667
[27] L-rel: 11.62500 / L-dis: 3.36062 / Best: 1.94648
 |- Curr: b" EFONG body\xce\x99\xce\x91\xce\xa3CA pope?$JR \xef\xbc\x89oration;) '');"
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 76963.50000
[28] L-rel: 11.56250 / L-dis: 2.10337 / Best: 1.94648
 |- Curr: b'icinaicinaorryGO pigerk_BLUE cor\xe6\xa0\xb9\xe6\x8d\xae measures})) conocer'
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 76835.83333
[29] L-rel: 11.56250 / L-dis: 2.72217 / Best: 1.94648
 |- Curr: b'\xe0\xb8\xb5\xe0\xb8\xa2\xe0\xb8\x81 \xe0\xa4\xb8\xe0\xa4\xb2_fl$reselse ConcePos armordeclar\xe2\x80\x9d?\xe9\xa0\x98()")\n'
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 76779.50000
[30] L-rel: 11.56250 / L-dis: 2.40577 / Best: 1.94648
 |- Curr: b"\xe3\x82\x89\xe3\x81\x84getStatus\xc3\xada Piet bombingstos_enheckpos eines')\xe5\x88\x92"
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 76765.50000
[31] L-rel: 11.56250 / L-dis: 2.16927 / Best: 1.94648
 |- Curr: b'_DAY watersUBE Peters unions\xe0\xb8\xad\xe0\xb8\xa7isValid flask \xc3\xb6imentos_with_for'
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 74521.50000
[32] L-rel: 11.43750 / L-dis: 3.06068 / Best: 1.94648
 |- Curr: b" Dependencies_TAB*dtOTT\xec\x83\x81 corpus.'); \xd0\xb8\xd0\xbc opioids)._parts od"
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 72275.25000
[33] L-rel: 11.37500 / L-dis: 2.69653 / Best: 1.94648
 |- Curr: b' charMontGOTWolvers Cassachu")]fst POR"),_Se'
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 70081.83333
[34] L-rel: 11.25000 / L-dis: 3.25875 / Best: 1.94648
 |- Curr: b" Cory plugs ImGuiimatACE BANK\xc3\xa9s reshapeuta \xe0\xb9\x82\xe0\xb8\x94\xe0\xb8\xa2']]);\n ];\n"
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 67958.91667
[35] L-rel: 11.18750 / L-dis: 2.37855 / Best: 1.94648
 |- Curr: b'ula\xc3\xa7\xc3\xa3oAuthorization\xeb\xb8\x8c Ley Mineralsclar/op)):());\n}], \xcf\x84\xce\xb7\xcf\x82"]:\n'
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 65926.33333
[36] L-rel: 11.12500 / L-dis: 3.27870 / Best: 1.94648
 |- Curr: b'(luaONEitation_RECORD Analysis Conce_connectionstops"].aciones}).\\")'
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 63979.16667
[37] L-rel: 11.06250 / L-dis: 3.47971 / Best: 1.94648
 |- Curr: b" ----------------------------------------------------------------ATTERY observationsodesSS gebruikCON sui arm.': h\xe1\xbb\xa3p teg"
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 62145.75000
[38] L-rel: 10.93750 / L-dis: 2.48525 / Best: 1.94648
 |- Curr: b'Enumeration \xd0\xb1\xd0\xbb\xd0\xb0\xd0\xb3 TAM\xd0\xba\xd1\x81 AUDicable)); Platz \xd0\xb7\xef\xbc\x89\xef\xbc\x9aacoes"":'
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 60425.33333
[39] L-rel: 10.87500 / L-dis: 2.02191 / Best: 1.94648
 |- Curr: b" Mex CONDITIONS\xe3\x83\x83\xe3\x83\x84 \xef\xbd\x9cchantslines\xef\xbf\xbd\xef\xbf\xbd substances \xd0\xbaEN\xc3\x8dadoras}';\n"
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 58808.16667
[40] L-rel: 10.81250 / L-dis: 2.83680 / Best: 1.94648
 |- Curr: b"rx probabilityajesDISmonitorStop \xd9\x83\xd8\xb3PLEMENT\xd1\x8b\xd0\xb2\xd0\xb0\xd0\xb5\xd1\x82KENUSE']\n"
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 57293.83333
[41] L-rel: 10.81250 / L-dis: 2.48136 / Best: 1.94648
 |- Curr: b'IEflexACSentiesFinULAokiesurs\tK \xcf\x84\xce\xb7\xce\xbd\xe4\xba\x8e\xe3\x82\xb5'
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 55875.33333
[42] L-rel: 10.68750 / L-dis: 3.09662 / Best: 1.94648
 |- Curr: b" saldo\xc3\x84 DESANGESudios Guide von unF\xc3\xbcr \xcf\x84\xcf\x89\xce\xbd')).\xe0\xb8\x8a"
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 54555.08333
[43] L-rel: 10.62500 / L-dis: 3.11708 / Best: 1.94648
 |- Curr: b'_getokerregistro library ces\xc2\xb6mutationsW>r\xc3\xa8mes)",\n \xc4\x91\xe1\xbb\x83'
 |- Best: b'\xe0\xb8\xb2\xe0\xb8\xa2\xe0\xb9\x83\xe0\xb8\x99 Town! TAB fistsictures BruLe\xe5\x9c\xa8ADOR\xef\xbc\x89\xe3\x80\x81 \xc3\xba'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 53341.00000
[44] L-rel: 10.62500 / L-dis: 1.80428 / Best: 1.80428
 |- Curr: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 52215.41667
[45] L-rel: 10.56250 / L-dis: 2.91635 / Best: 1.80428
 |- Curr: b' Initialize$rows Lines\xd0\xbe\xd1\x8eections rozslaps\xef\xbf\xbd Officials belts]";\n \xd0\xbe\xd1\x81'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 51181.83333
[46] L-rel: 10.50000 / L-dis: 2.35841 / Best: 1.80428
 |- Curr: b'++ cassGSMARYRESSpare\xc3\xa9scurrentUser\xe0\xb8\xaaniciORS "),\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 50232.25000
[47] L-rel: 10.43750 / L-dis: 3.09926 / Best: 1.80428
 |- Curr: b" we closeButtonEXT_cache_current GOODS timestampSelectorsoration Ces seja'))\n"
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 49360.66667
[48] L-rel: 10.37500 / L-dis: 2.32070 / Best: 1.80428
 |- Curr: b' Gross \xe7\xbb\x93isease mikbacks_ly permissiblepositions_past")}\n\xc2\xbb,]")\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 48559.66667
[49] L-rel: 10.37500 / L-dis: 3.18967 / Best: 1.80428
 |- Curr: b' NextSA =====\xec\x9e\x84\xe0\xa4\xbe\xe0\xa4\x9c_annotations underscores neckcas \xd0\x9f\xd0\xb5\xd1\x80()); p\xc3\xbab'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 47825.75000
[50] L-rel: 10.31250 / L-dis: 2.71139 / Best: 1.80428
 |- Curr: b"_push303 Cont Libc\xc4\xb1s\xc4\xb1 Branchbrowsernotification strands']:\xef\xbc\x9a});"
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 47155.83333
[51] L-rel: 10.25000 / L-dis: 2.93848 / Best: 1.80428
 |- Curr: b' AutoFileStream\xcf\x81\xcf\x8c\xcf\x82Clustercrets Shutterstock)\': substances">% \xce\xba\xce\xb1\xe1\xbd\xb6\xe3\x80\x8b\xef\xbc\x8c"]\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 46543.75000
[52] L-rel: 10.25000 / L-dis: 2.53955 / Best: 1.80428
 |- Curr: b' Se\xe3\x82\xb8\xe3\x83\xa5\xd8\xb2\xd9\x85\xd8\xa7\xd9\x86\xd0\xb5\xd0\xbb\xd1\x8c/sh segurouluije)):"]).\xc3\xa1tu}")'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 45987.33333
[53] L-rel: 10.18750 / L-dis: 2.69105 / Best: 1.80428
 |- Curr: b'      unprotected\xeb\xa7\x81 \xce\xbc\xcf\x8c\xce\xbd\xce\xbfDescriptorsKE\xc5\xa5 masksAPH \xcf\x84\xce\xbf\xcf\x85\xc2\xb4:ui\xc3\xa7\xc3\xa3o'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 45485.50000
[54] L-rel: 10.12500 / L-dis: 2.65920 / Best: 1.80428
 |- Curr: b',\n procent157.toFloat\xeb\xb8\x8c.check/re apparatus \xe3\x83\x96 ! za\xe3\x80\x82\xe5\x9c\xa8'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 45033.41667
[55] L-rel: 10.12500 / L-dis: 2.34576 / Best: 1.80428
 |- Curr: b' shellSTEP_valismu kissrasespartment__:()))\n\xea\xb8\x88ANCES\xc5\xa0'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 44628.91667
[56] L-rel: 10.12500 / L-dis: 2.84093 / Best: 1.80428
 |- Curr: b' LinesSEL NOTES_ianness_bankoints approve<?>>\xe6\xa0\xb9\xe6\x8d\xae\\")/)\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 44271.83333
[57] L-rel: 10.12500 / L-dis: 2.59605 / Best: 1.80428
 |- Curr: b'\xe0\xa4\xbf\xe0\xa4\xa3ChecksOps {?} bus anlay\xc4\xb1\xc5\x9f\xd1\x81\xd1\x82\xd1\x8b\xd8\xa8\xd8\xb1\xdb\x8c \xc5\xa1tcasov\xc3\xa1ny]));\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 43960.83333
[58] L-rel: 10.06250 / L-dis: 2.97981 / Best: 1.80428
 |- Curr: b' !Use.isOpenCache\xe0\xa4\xb8fgAccessType impossvert]];)\xeb\xa5\xbc"]))\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 43692.58333
[59] L-rel: 10.06250 / L-dis: 2.55341 / Best: 1.80428
 |- Curr: b' Guarantee\xd1\x96\xd1\x80atriceolds legislatorssym \xd9\x81\xdb\x8c possibile dellakening nelle\xe0\xb8\x9b'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 43462.25000
[60] L-rel: 10.06250 / L-dis: 2.75985 / Best: 1.80428
 |- Curr: b'NG shell\tsysasse\xe3\x81\xaeDataSourceraries\xe3\x83\xaa\xe3\x82\xa2\xe8\xba\xab\xe4\xbd\x93_K_on")\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 43268.91667
[61] L-rel: 10.00000 / L-dis: 2.72008 / Best: 1.80428
 |- Curr: b' it corazMessenger guidelines tree_threads siguientesgetCurrent\xd0\xb2\xd0\xb0\xd0\xb5\xd1\x82]";\n\xef\xbc\x8c\xe5\x9c\xa8 ]'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01001
 |- Avg Top P-99: 43109.66667
[62] L-rel: 10.00000 / L-dis: 2.86913 / Best: 1.80428
 |- Curr: b"leetcode Practicescommunicationobjective\xeb\xb3\x91\xc2\xa0\xd0\xbf ! ty_f']);\xce\xba\xce\xb1\xcf\x82 \xcf\x84\xce\xb7\xcf\x82"
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 42980.83333
[63] L-rel: 10.00000 / L-dis: 2.59531 / Best: 1.80428
 |- Curr: b' unUselibrariesvariationfigures\xe2\x80\x82ufs\xef\xbc\x8cjurylays \xce\xb1\xcf\x80\xcf\x8c";}\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 42878.75000
[64] L-rel: 10.00000 / L-dis: 2.51379 / Best: 1.80428
 |- Curr: b' {ettesfectionplitsGUdress_connectionannual \xd7\x94));\xef\xbf\xbd://'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 42800.91667
[65] L-rel: 10.00000 / L-dis: 2.07979 / Best: 1.80428
 |- Curr: b' outer_DATE SHOPutersVASwa/sw \xc5\x9b552\xe8\xaf\x81\xe6\x98\x8e\xce\x99\xce\x91\xce\xa3\xef\xbc\x8c\xe4\xb8\x8e'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 42744.25000
[66] L-rel: 10.00000 / L-dis: 2.90298 / Best: 1.80428
 |- Curr: b'_TIMEOUT\xef\xbf\xbd\xe6\x96\xadgetStatusbuttonsdays\xe3\x82\xb5 d\xe1\xbb\xabng\xd0\xba\xd1\x96\xd0\xb2clarations \xd0\xa1}):(""))\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 42705.25000
[67] L-rel: 10.00000 / L-dis: 3.29773 / Best: 1.80428
 |- Curr: b' shell Enemy\tsys\xd1\x83\xd0\xb7_pos\xcf\x80\xce\xb1 Registers dir\xc3\xa1l))._k])),\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 42680.50000
[68] L-rel: 10.00000 / L-dis: 2.34665 / Best: 1.80428
 |- Curr: b" TourismRelativeTo \xd1\x86\xd1\x96\xd1\x94\xd1\x97rxCadastro repospling \xec\x8a\xa4\xe9\x98\xb2']).adoras]>"
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 42666.83333
[69] L-rel: 10.00000 / L-dis: 2.65913 / Best: 1.80428
 |- Curr: b"unkt sockShell=sENCY Assistance=$(ictionaries_on']).\xe0\xb9\x87\xe0\xb8\x99\xe0\xb8\xaa]]."
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 42660.83333
[70] L-rel: 9.93750 / L-dis: 3.16243 / Best: 1.80428
 |- Curr: b' la<Transformicrosidersitions_backend?$kf]").\xe3\x81\xab\xe3\x81\xa4 \xd0\xb7\xd0\xb0")}'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 42659.33333
[71] L-rel: 10.00000 / L-dis: 2.67692 / Best: 1.80428
 |- Curr: b'Kingroys ########Changes Drugslights !__:beltjury.")]\n\xef\xbc\x81'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 41621.75000
[72] L-rel: 9.87500 / L-dis: 2.80629 / Best: 1.80428
 |- Curr: b' currentPage Knowledge \xd0\xbe\xd1\x81\xd0\xbe\xd0\xb1\xd0\xb5\xd0\xbd\xd0\xbd\xd0\xbe.isUser/tokenudes\xe0\xb9\x8c\xe0\xb9\x81\xe0\xb8\xa5\xe0\xb8\xb0 onay_NOTopiasures")\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 40486.25000
[73] L-rel: 9.81250 / L-dis: 3.28709 / Best: 1.80428
 |- Curr: b'(*( JACKetas cases\xe8\x90\xa8 reconstruction\xe6\x9d\xbe\xe6\x8c\x89\xe7\x85\xa7probC.")]\n!),'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 39304.75000
[74] L-rel: 9.75000 / L-dis: 3.24223 / Best: 1.80428
 |- Curr: b' AutoemoryEQUAL_Release_possible ces/__]];()))\n\']"). \xd7\x91}`\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 38101.08333
[75] L-rel: 9.68750 / L-dis: 2.22491 / Best: 1.80428
 |- Curr: b' GDPR Changesroys_pos\xd0\xb0\xd0\xbd\xd0\xb8\xd1\x8eqxlej limits dalle )./fs \xd0\xad'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 36914.50000
[76] L-rel: 9.62500 / L-dis: 2.37906 / Best: 1.80428
 |- Curr: b"########\xd8\xaa\xd8\xa8\xd8\xb1esses<Resourceensingpace warnings fore le    ');)}\n"
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 35751.25000
[77] L-rel: 9.56250 / L-dis: 2.55539 / Best: 1.80428
 |- Curr: b" U \xd8\xa8\xd8\xb1\xda\xaf \xea\xb7\x9c.object\xd1\x81\xd1\x82\xd0\xb8connections\xe6\xb3\x95\xef\xbf\xbd'),\xc3\xa4ttoks p\xc3\xa4"
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 34627.83333
[78] L-rel: 9.43750 / L-dis: 2.50212 / Best: 1.80428
 |- Curr: b' on_yes considSES damaging \xd8\xb1\xd8\xacreopen\xd0\xb7\xe7\x9a\x84\xe6\x98\xaf \xce\xba\xce\xb1\xe1\xbd\xb6"": \xe0\xb8\xa7'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 33545.41667
[79] L-rel: 9.37500 / L-dis: 2.48515 / Best: 1.80428
 |- Curr: b'TS \xe3\x83\xac/logsance-\xd0\xa1\xe5\xa3\xablayoutModification:r:** \xcf\x80\xce\xbf\xcf\x85."),\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.01000
 |- Avg Top P-99: 32515.91667
[80] L-rel: 9.37500 / L-dis: 2.65611 / Best: 1.80428
 |- Curr: b'\xe6\x96\xb0\xe7\x9a\x84_resource procentnexmutations_dyastsurs\xec\x9d\x8c\xec\x9d\x84 prz\')"\n"],\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00999
 |- Avg Top P-99: 31542.00000
[81] L-rel: 9.31250 / L-dis: 2.75688 / Best: 1.80428
 |- Curr: b'Nextimeters.bus Forces\xce\x91\xce\x99_bsusz \xd0\xa1ViewsCAS")}\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00999
 |- Avg Top P-99: 30626.25000
[82] L-rel: 9.25000 / L-dis: 2.72498 / Best: 1.80428
 |- Curr: b'(_.########grantDates t\xc6\xb0\xcf\x80\xce\xb1Checker \xef\xbf\xbd\']/\xe0\xb9\x87\xe0\xb8\x99\xe0\xb8\xaa \xc5\x9b:";\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00999
 |- Avg Top P-99: 29762.50000
[83] L-rel: 9.18750 / L-dis: 2.49570 / Best: 1.80428
 |- Curr: b'IME vhodn\xc3\xa9\xea\xb8\xb0\xea\xb0\x80\xc4\x9bt\xc3\xadengerseresasses at],\tSzas ?>>'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00999
 |- Avg Top P-99: 28952.58333
[84] L-rel: 9.18750 / L-dis: 3.07027 / Best: 1.80428
 |- Curr: b".Open_CONTINUERESParkersAnalysisbrasmiss flips()._ delle_partner']))\n"
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00999
 |- Avg Top P-99: 28193.16667
[85] L-rel: 9.12500 / L-dis: 2.72591 / Best: 1.80428
 |- Curr: b' alcuni_yes tails NazisallisMask Possible Einsatz_kPodies")}\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00999
 |- Avg Top P-99: 27475.41667
[86] L-rel: 9.06250 / L-dis: 2.83671 / Best: 1.80428
 |- Curr: b' "$Trees Nazis\xc5\xbc vets\xe2\x80\x82uplicates__:()].P\']]);\n()}\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00999
 |- Avg Top P-99: 26807.66667
[87] L-rel: 9.00000 / L-dis: 2.90930 / Best: 1.80428
 |- Curr: b'`ierung\xe9\x96\xa2\xe4\xbf\x82_inputsensingentries_guardanks \xef\xbc\x9a\xe0\xb8\xaa ces,),\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00999
 |- Avg Top P-99: 26177.25000
[88] L-rel: 8.93750 / L-dis: 2.80977 / Best: 1.80428
 |- Curr: b" consider_lex timespec LegacyGameState_pipemarks stmt:s \xd7\x94}).'),\n"
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00999
 |- Avg Top P-99: 25576.08333
[89] L-rel: 8.93750 / L-dis: 2.44743 / Best: 1.80428
 |- Curr: b'ierungflex"".CSSequalityUnsafe chest limit \xd0\xa0**:\xeb\xa6\xac\xeb\xa5\xbc});'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00998
 |- Avg Top P-99: 25007.58333
[90] L-rel: 8.87500 / L-dis: 2.73770 / Best: 1.80428
 |- Curr: b"ursion_spin \xd0\xbb\xd1\x96 weapons p\xc5\x99itom\xe9\x96\x80\xc3\xadnyplies';armsvas\xe3\x80\x8d"
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00998
 |- Avg Top P-99: 24472.66667
[91] L-rel: 8.87500 / L-dis: 2.47394 / Best: 1.80428
 |- Curr: b'>(\n \xd0\xb2\xd0\xb8\xd0\xba\xd0\xbe\xd0\xbd\xd0\xb0\xd0\xbd\xd0\xbd\xd1\x8f_mtxallisimeobj\xe6\xb3\x95Fore\xc2\xbb.`:\xe7\x9a\x84\xe6\x98\xaf ),\r\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00998
 |- Avg Top P-99: 23972.08333
[92] L-rel: 8.81250 / L-dis: 2.37821 / Best: 1.80428
 |- Curr: b'ukturistringgue \xd0\xbb\xd1\x96\xd0\xba\xe8\x90\xa8Risk\xd1\x8e\xd1\x89\xd0\xb5\xd0\xb9\xe7\xae\xa1), {}).\\")();}\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00998
 |- Avg Top P-99: 23501.91667
[93] L-rel: 8.81250 / L-dis: 2.39409 / Best: 1.80428
 |- Curr: b'=re\xe2\x80\x82.help\\Controllersns_SAFEuplicates__:\xd1\x94\xd1\x8epiar\xe2\x94\x81\xe2\x94\x81\xe2\x94\x81\xe2\x94\x81\xe2\x94\x81\xe2\x94\x81\xe2\x94\x81\xe2\x94\x81)";\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00998
 |- Avg Top P-99: 23062.16667
[94] L-rel: 8.75000 / L-dis: 2.80372 / Best: 1.80428
 |- Curr: b' vacation.utils modificationveculsesprevent\xc3\x81 caches dalle\xe2\x80\x99.\xe3\x82\x92()}\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00998
 |- Avg Top P-99: 22649.41667
[95] L-rel: 8.75000 / L-dis: 2.52858 / Best: 1.80428
 |- Curr: b'Parser Neck \xea\xb3\x84\xec\x95\xbd \xd1\x82\xd0\xb0\xd0\xba\xd0\xbe\xd0\xb5/es\xe9\x96\x80 c\xc3\xad\xc3\xa1rios \xd0\x97\'] ),\r\n\']").'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00998
 |- Avg Top P-99: 22261.08333
[96] L-rel: 8.68750 / L-dis: 2.19295 / Best: 1.80428
 |- Curr: b'Selector Eggs\xec\x9e\x90\xea\xb0\x80imatorsneckcolorsFS\xef\xbf\xbdTIMvy \xd9\x84\xd8\xb3\xeb\xa6\xac\xeb\xa5\xbc'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00998
 |- Avg Top P-99: 21896.08333
[97] L-rel: 8.62500 / L-dis: 2.30070 / Best: 1.80428
 |- Curr: b" bring.':afs inters POSSIBILITY\xe0\xb8\xa5\xe0\xb8\x87\xe1\xba\xa3ngovers oleh\xe2\x80\x99:\xec\x9e\x90\xeb\xa5\xbc>(),\n"
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00998
 |- Avg Top P-99: 21553.75000
[98] L-rel: 8.62500 / L-dis: 2.93241 / Best: 1.80428
 |- Curr: b'(debugExtensionsigers\xd1\x86\xd0\xb8\xd0\xbaellasys ries documents\xd1\x92 por`}\')";\n'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00997
 |- Avg Top P-99: 21233.75000
[99] L-rel: 8.56250 / L-dis: 3.06262 / Best: 1.80428
 |- Curr: b'.is\xe5\xba\x8fcis \xd8\xb4\xd9\x86\xd8\xa7\xd8\xb3-\xd0\xa1uebas\xe0\xa4\xb7\xe4\xba\x8e\xd0\xba\xd0\xb0\xd1\x81%), regulators\xef\xbc\x89\xef\xbc\x9a'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00997
 |- Avg Top P-99: 20932.00000
[100] L-rel: 8.56250 / L-dis: 2.76691 / Best: 1.80428
 |- Curr: b',\n_pairs bosses Determine Guides/fsNotifications\xc3\xbdchuces\xe3\x81\xae\xe3\x82\x92});'
 |- Best: b'SafetyONESgetFile\xd9\x85\xd9\x88\xd8\xafriskLetters vypariesiscopal \xce\xba\xce\xb1\xce\xb9_slices \xd0\x91'
 |- Avg Max Prob: 0.00997
 |- Avg Top P-99: 20648.66667

[+] Done. Final loss: 1.8042826652526855
"""

# Calculate metrics
metrics = parse_log(log_data)

# Print metrics
for key, value in metrics.items():
    print(f"{key}: {value}")