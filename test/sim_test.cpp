#include "embedding.h"
#include <iostream>

std::vector<float> str2vec(const std::string &s) {
    std::vector<std::string> words;
    words.push_back(s);
    std::string joined                   = join(words, "\n");
    std::vector<std::vector<float>> vecs = embedding(joined);
    return vecs[0];
}

int main()
{
    std::string query = "The low mass of the planet has led to suggestions that it may be a terrestrial planet. This type of massive terrestrial planet could be formed in the inner part of the Gliese 876 system from material pushed towards the star by the inward migration of the gas giants.";
    std::string str1  = "Before the Grand Louvre overhaul of the late 1980s and 1990s, the Louvre had several street-level entrances, most of which are now permanently closed. Since 1993, the museum's main entrance has been the underground space under the Louvre Pyramid, or Hall Napoléon, which can be accessed from the Pyramid itself, from the underground Carrousel du Louvre, or (for authorized visitors) from the passage Richelieu connecting to the nearby rue de Rivoli. A secondary entrance at the Porte des Lions, near the western end of the Denon Wing, was created in 1999 but is not permanently open.";
    std::string str2  = "The Louvre museum is located inside the Louvre Palace, in the center of Paris, adjacent to the Tuileries Gardens. The two nearest Métro stations are Louvre-Rivoli and Palais Royal-Musée du Louvre, the latter having a direct underground access to the Carrousel du Louvre commercial mall."; 

    std::vector<float> query_vec = str2vec(query);
    std::vector<float> str1_vec  = str2vec(str1);
    std::vector<float> str2_vec  = str2vec(str2);

    size_t n_embd = query_vec.size();
    float sim1 = common_embd_similarity_cos(query_vec.data(), str1_vec.data(), n_embd);
    float sim2 = common_embd_similarity_cos(query_vec.data(), str2_vec.data(), n_embd);

    std::cout << "Similarity with str1: " << sim1 << std::endl;
    std::cout << "Similarity with str2: " << sim2 << std::endl;

    return 0;
}