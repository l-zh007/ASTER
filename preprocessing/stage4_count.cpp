#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <zlib.h>

namespace {

constexpr int kBins = 100;
constexpr int kContexts = 8;
constexpr int kHalfWindow = 10000;
constexpr int kBinWidth = 200;

struct CellInfo {
  uint16_t context;
  uint16_t local_index;
};

uint32_t EncodeBarcode(const char* begin, const char* end) {
  uint32_t value = 0;
  int length = 0;
  for (const char* p = begin; p < end && *p != '-'; ++p) {
    value <<= 2;
    switch (*p) {
      case 'A': break;
      case 'C': value |= 1U; break;
      case 'G': value |= 2U; break;
      case 'T': value |= 3U; break;
      default: throw std::runtime_error("Invalid 10x barcode");
    }
    ++length;
  }
  if (length != 16) throw std::runtime_error("Expected a 16-base 10x barcode");
  return value;
}

int ParseChrom(const char* begin, const char* end) {
  if (end - begin < 4 || begin[0] != 'c' || begin[1] != 'h' || begin[2] != 'r') return 0;
  int chrom = 0;
  for (const char* p = begin + 3; p < end; ++p) {
    if (*p < '0' || *p > '9') return 0;
    chrom = chrom * 10 + (*p - '0');
  }
  return chrom >= 1 && chrom <= 22 ? chrom : 0;
}

bool InBlacklist(const std::vector<std::pair<int, int>>& intervals, int position) {
  auto it = std::upper_bound(
      intervals.begin(), intervals.end(), position,
      [](int value, const std::pair<int, int>& interval) { return value < interval.first; });
  if (it == intervals.begin()) return false;
  --it;
  return position >= it->first && position < it->second;
}

void WriteBinary(const std::string& path, const std::vector<uint32_t>& values) {
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("Cannot create output: " + path);
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(uint32_t)));
  if (!output) throw std::runtime_error("Failed while writing: " + path);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: stage4_count variants.tsv cells.tsv blacklist.bed fragments.tsv.gz "
                 "counts.bin support.bin stats.tsv\n";
    return 2;
  }

  try {
    std::array<std::vector<std::pair<int, uint32_t>>, 23> variants;
    uint32_t n_variants = 0;
    {
      std::ifstream input(argv[1]);
      int chrom = 0, pos0 = 0;
      uint32_t row = 0;
      while (input >> chrom >> pos0 >> row) {
        if (chrom < 1 || chrom > 22) throw std::runtime_error("Invalid variant chromosome");
        variants[chrom].emplace_back(pos0, row);
        n_variants = std::max(n_variants, row + 1);
      }
      for (int chrom = 1; chrom <= 22; ++chrom) std::sort(variants[chrom].begin(), variants[chrom].end());
    }

    std::unordered_map<uint32_t, CellInfo> cells;
    std::array<uint32_t, kContexts> n_cells{};
    {
      std::ifstream input(argv[2]);
      std::string barcode;
      int context = 0, local_index = 0;
      while (input >> barcode >> context >> local_index) {
        if (context < 0 || context >= kContexts || local_index < 0 || local_index >= 65536)
          throw std::runtime_error("Invalid cell map");
        uint32_t key = EncodeBarcode(barcode.data(), barcode.data() + barcode.size());
        if (!cells.emplace(key, CellInfo{static_cast<uint16_t>(context),
                                         static_cast<uint16_t>(local_index)}).second)
          throw std::runtime_error("Duplicate barcode within donor");
        ++n_cells[context];
      }
    }

    std::array<std::vector<std::pair<int, int>>, 23> blacklist;
    {
      std::ifstream input(argv[3]);
      std::string chromosome;
      int start = 0, end = 0;
      while (input >> chromosome >> start >> end) {
        int chrom = ParseChrom(chromosome.data(), chromosome.data() + chromosome.size());
        if (chrom) blacklist[chrom].emplace_back(start, end);
      }
      for (int chrom = 1; chrom <= 22; ++chrom) std::sort(blacklist[chrom].begin(), blacklist[chrom].end());
    }

    const uint64_t n_values = static_cast<uint64_t>(n_variants) * kContexts * kBins;
    std::vector<uint32_t> counts(n_values, 0U);
    std::vector<uint32_t> support(n_values, 0U);
    std::array<uint64_t, kContexts> valid_cuts{};
    std::vector<uint64_t> support_events;
    support_events.reserve(1U << 24);

    auto flush_support = [&]() {
      std::sort(support_events.begin(), support_events.end());
      auto last = std::unique(support_events.begin(), support_events.end());
      for (auto it = support_events.begin(); it != last; ++it) {
        uint64_t flat = *it >> 16;
        ++support[flat];
      }
      support_events.clear();
    };

    auto add_cut = [&](int chrom, int position, const CellInfo& cell) {
      if (InBlacklist(blacklist[chrom], position)) return;
      ++valid_cuts[cell.context];
      const auto& chr_variants = variants[chrom];
      auto first = std::lower_bound(
          chr_variants.begin(), chr_variants.end(), position - kHalfWindow + 1,
          [](const std::pair<int, uint32_t>& variant, int value) { return variant.first < value; });
      auto last = std::upper_bound(
          chr_variants.begin(), chr_variants.end(), position + kHalfWindow,
          [](int value, const std::pair<int, uint32_t>& variant) { return value < variant.first; });
      for (auto it = first; it != last; ++it) {
        const int bin = (position - (it->first - kHalfWindow)) / kBinWidth;
        if (bin < 0 || bin >= kBins) continue;
        const uint64_t flat =
            (static_cast<uint64_t>(it->second) * kContexts + cell.context) * kBins + bin;
        ++counts[flat];
        support_events.push_back((flat << 16) | cell.local_index);
      }
    };

    gzFile input = gzopen(argv[4], "rb");
    if (!input) throw std::runtime_error("Cannot open fragments file");
    std::array<char, 1024> line{};
    int previous_chrom = 0;
    uint64_t selected_fragments = 0;
    while (gzgets(input, line.data(), static_cast<int>(line.size())) != nullptr) {
      if (line[0] == '#') continue;
      char* tab1 = std::find(line.data(), line.data() + line.size(), '\t');
      if (tab1 == line.data() + line.size()) continue;
      char* tab2 = std::find(tab1 + 1, line.data() + line.size(), '\t');
      char* tab3 = std::find(tab2 + 1, line.data() + line.size(), '\t');
      char* tab4 = std::find(tab3 + 1, line.data() + line.size(), '\t');
      if (tab2 == line.data() + line.size() || tab3 == line.data() + line.size() ||
          tab4 == line.data() + line.size()) continue;
      const int chrom = ParseChrom(line.data(), tab1);
      if (!chrom) continue;
      if (previous_chrom != 0 && chrom != previous_chrom) flush_support();
      previous_chrom = chrom;

      uint32_t barcode = EncodeBarcode(tab3 + 1, tab4);
      auto cell = cells.find(barcode);
      if (cell == cells.end()) continue;
      const int start = static_cast<int>(std::strtol(tab1 + 1, nullptr, 10));
      const int end = static_cast<int>(std::strtol(tab2 + 1, nullptr, 10));
      if (end <= start) continue;
      ++selected_fragments;
      add_cut(chrom, start, cell->second);
      add_cut(chrom, end - 1, cell->second);
    }
    gzclose(input);
    flush_support();

    WriteBinary(argv[5], counts);
    WriteBinary(argv[6], support);
    std::ofstream stats(argv[7]);
    stats << "context_id\tn_cells\ttotal_valid_cuts\n";
    for (int context = 0; context < kContexts; ++context)
      stats << context << '\t' << n_cells[context] << '\t' << valid_cuts[context] << '\n';
    std::cerr << "selected_fragments=" << selected_fragments << "\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
  return 0;
}
