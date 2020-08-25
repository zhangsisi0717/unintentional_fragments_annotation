# mz.cloud Part

from pathlib import Path

save_path = Path('../../../lab_meeting/07_17_2020_labmeeting/pics')
assert save_path.is_dir()

from ms_data_workflow import *
from mz_cloud_workflow import *

m.base_cos_similarity(plot=True)
plt.tight_layout()
plt.savefig((save_path/"3T3_base_cos.png").as_posix(), dpi=300, transparent=True)
plt.show()

# base_iter = iter(sorted(enumerate(r.n_components for r in m.base_info.values()),
#                         key=lambda x: x[1], reverse=True))

base_iter = iter(sorted(enumerate(r.v_max for r in m.base_info.values()),
                        key=lambda x: x[1], reverse=True))

# target =
# i = np.argmin(np.abs([b.rt - target for b in m.base_info.values()]))
# m.base_info[m.base_index[i]], m.base_info[m.base_index[i]].rt

i, _ = next(base_iter)


base = m.base_info[m.base_index[i]]

m.gen_spectrum(i, plot=True, load=False, **spec_params)
plt.show()
#
m.plot_coelution(i, load=False, rescale=True, **spec_params)
plt.show()
# # #
# m.plot_coelution(i, rescale=False, **spec_params)
# plt.show()

matched = mzc.find_match(base.spectrum, threshold=1E-3, transform=math.sqrt)
print(i, matched[0][0].CompoundName, matched[0][2])
base.spectrum.compare_spectrum_plot(matched[0][1])
plt.show()

plt.tight_layout()
plt.savefig((save_path/"3T3_L-Glutamic_acid.png").as_posix(), dpi=300, transparent=True)
plt.show()


for c, s, cos in matched[:10]:
    print("{}\t{:.4f}".format(c.CompoundName, cos))

c = matched[9][0]
it = (s for t in c.spectra_1 + c.spectra_2 for s in t if s.Polarity == 'Negative')

max = 0.
argmax = None
for s in it:
    cos = s.bin_vec.cos(base.spectrum.bin_vec)
    if cos > max:
        max = cos
        argmax = s

base.spectrum.compare_spectrum_plot(argmax)
plt.tight_layout()
plt.savefig((save_path/"rw_standards_glucose_spurious_matched.png").as_posix(), dpi=300, transparent=True)
plt.show()

for c, s, cos in matched[:10]:
    print("{}\t{:.4f}".format(c.CompoundName, cos))


s = next(it)
cos = s.bin_vec.cos(base.spectrum.bin_vec)
s.compare_spectrum_plot(base.spectrum)
plt.title("cos = {:.4f}".format(cos))
plt.show()

log_base = matched[7][1].bin_vec.log_base

print("\n".join("{:.5f}".format(i) for i in matched[7][1].mz[matched[7][1].relative_intensity >= 0.05]))
print("\n".join("{}".format(int(np.floor(np.log(i) / log_base))) for i in matched[7][1].mz[matched[7][1].relative_intensity >= 0.05]))

print("\n".join("{:.5f}".format(i) for i in matched[7][1].relative_intensity[matched[7][1].relative_intensity >= 0.05]))


print("\n".join("{:.5f}".format(i) for i in base.spectrum.mz[base.spectrum.relative_intensity >= 0.05]))
print("\n".join("{}".format(int(np.floor(np.log(i) / log_base))) for i in base.spectrum.mz[base.spectrum.relative_intensity >= 0.05]))

print("\n".join("{:.5f}".format(i) for i in base.spectrum.relative_intensity[base.spectrum.relative_intensity >= 0.05]))