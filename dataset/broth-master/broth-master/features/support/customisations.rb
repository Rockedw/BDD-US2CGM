
fixtures_folder = File.join(RAILS_ROOT, 'spec', 'fixtures')
Fixtures.create_fixtures(fixtures_folder, %w[site_settings])

require 'pickle/world'
# Example of configuring pickle:
#
# Pickle.configure do |config|
#   config.adapters = [:machinist]
#   config.map 'I', 'myself', 'me', 'my', :to => 'user: "me"'
# end
