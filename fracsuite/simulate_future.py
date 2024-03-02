
# def stoch_prob_test(
#     name: str,
#     sigma_s: float = None,
#     thickness: float = None,
#     boundary: SpecimenBoundary = None,
#     break_pos: SpecimenBreakPosition = None,
#     no_region_crop: bool = False,
#     validate: int = -1
# ):
#     """Validate alfa by running the algorithm `validate` times and plotting the cdf onto the specimen cdf."""
#     specimen = Specimen.get(name)

#     if sigma_s is None:
#         sigma_s = np.abs(specimen.sig_h)
#     if thickness is None:
#         thickness = specimen.thickness
#     size = specimen.get_real_size()
#     if boundary is None:
#         boundary = specimen.boundary
#     if break_pos is None:
#         break_pos = specimen.break_pos
#     E = 70e3
#     nue = 0.23

#     info(f'Creating simulation for {specimen.name}')
#     info(f'> Sigma_s: {sigma_s}')
#     info(f'> Thickness: {thickness} (real: {specimen.measured_thickness})')
#     info(f'> Size: {size}')
#     info(f'> Boundary: {boundary}')
#     info(f'> Break position: {break_pos}')

#     if validate == -1:
#         # create simulation
#         simulation = alfa(sigma_s, thickness, size, boundary, break_pos, E, nue, impact_position=specimen.get_impact_position(),
#                             no_region_crop=no_region_crop, reference=specimen.name)
#         # compare simulation with input
#         compare(simulation.fullname, specimen.name)

#         # put the original fracture image on a figure into the simulatio
#         fig0,axs0 = plt.subplots(figsize=get_fig_width(FigureSize.ROW1))
#         axs0.imshow(specimen.get_fracture_image())
#         axs0.set_xlabel('x (px)')
#         axs0.set_ylabel('y (px)')
#         axs0.grid(False)
#         fig0.savefig(simulation.get_file('original_fracture_image.pdf'))

#         return simulation
#     else:
#         spec_areas = [s.area for s in specimen.splinters]
#         binrange = get_log_range(spec_areas, 30)

#         fig0,axs0 = None,None
#         with plt.ion():
#             fig0,axs0 = datahist_plot(figwidth=FigureSize.ROW1)
#             datahist_to_ax(axs0, spec_areas, binrange=binrange, color='C0', data_mode=DataHistMode.CDF)
#             fig0.canvas.draw()
#             fig0.canvas.flush_events()

#         for i in range(validate):
#             sim = alfa(sigma_s, thickness, size, boundary, break_pos, E, nue, impact_position=specimen.get_impact_position(),
#                             no_region_crop=no_region_crop, reference=specimen.name)
#             areas = [s.area for s in sim.splinters]
#             info(f'Validation {i+1}/{validate}: {len(sim.splinters)} splinters, mean area: {np.mean(areas):.2f} mm²')
#             datahist_to_ax(axs0, areas, binrange=binrange, color='C1', data_mode=DataHistMode.CDF, linewidth=0.3)
#             fig0.canvas.draw()
#             fig0.canvas.flush_events()

#         axs0[0].set_xlabel('Bruchstückflächeninhalt $A_\mathrm{S}$ (mm²)')
#         axs0[0].set_ylabel('CDF')
#         legend_without_duplicate_labels(axs0[0])
#         State.output(fig0, f'validation_{specimen.name}', figwidth=FigureSize.ROW1)
